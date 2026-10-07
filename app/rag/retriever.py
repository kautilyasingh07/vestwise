"""Policy retrieval: `$vectorSearch` with the access pre-filter and a score threshold,
plus BM25 keyword search merged by reciprocal rank fusion (spec §8.2).

All of `company_id`, `role` and `stakeholder_id` come from the server-side
request context, never from the LLM. The same `build_access_filter` result
restricts **both** retrievers: it is the `$vectorSearch` pre-filter, and it is
the `find()` query that loads the BM25 corpus. The BM25 index is built per
request from that corpus only, so a forbidden chunk is never a candidate in
either list (and never contributes to BM25's IDF statistics).

Hybrid flow: vector top FUSION_DEPTH -> drop cosine < min_score -> if nothing
is left, return [] ("not found", no BM25, no LLM) -> BM25 top FUSION_DEPTH over
the user's corpus -> RRF -> top k. BM25-only chunks get their cosine computed
locally from the stored embedding, so `score` is always a cosine.

Scores are returned as **cosine similarity**. Atlas reports
`vectorSearchScore = (1 + cosine) / 2` for a cosine index, so a raw Atlas score
of 0.71 is a cosine of 0.42; we convert back before thresholding, otherwise the
spec's cosine threshold would be compared against the wrong scale.

Mongo, config and the embedding model are imported lazily inside `retrieve`,
so the pure helpers here can be tested without `.env` (same pattern as the chunker).
"""

from dataclasses import dataclass, replace
from typing import Any, Literal

from app.rag.access import Role, build_access_filter, is_visible
from app.rag.hybrid import RRF_K, bm25_ranking, rrf_fuse

NUM_CANDIDATES = 100  # spec §8.2: HNSW candidate pool; `limit` (k) is taken from it
FUSION_DEPTH = 20  # each retriever contributes its top 20 to RRF

RetrievalMode = Literal["vector", "hybrid"]

# Fields loaded for the BM25 corpus: what a RetrievedChunk needs, the two access
# fields (re-checked in Python), and the embedding (cosine for BM25-only hits).
CORPUS_PROJECTION: dict[str, Any] = {
    "_id": 1, "company_id": 1, "owner_stakeholder_id": 1, "doc_id": 1, "doc_title": 1,
    "doc_type": 1, "page": 1, "section": 1, "text": 1, "embedding": 1,
}

# Fields returned to the caller; `embedding` is never projected out of the database.
PROJECTION: dict[str, Any] = {
    "_id": 1,
    "doc_id": 1,
    "doc_title": 1,
    "doc_type": 1,
    "page": 1,
    "section": 1,
    "text": 1,
    "owner_stakeholder_id": 1,
    "score": {"$meta": "vectorSearchScore"},
}


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit, with what a citation needs (`[doc_title, p. page]`)."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    page: int
    section: str
    text: str
    score: float  # cosine similarity, -1..1
    owner_stakeholder_id: str | None
    vector_rank: int | None = None  # 1-based rank in each retriever's list; None = not in it
    bm25_rank: int | None = None
    rrf_score: float | None = None  # set in hybrid mode


def atlas_score_to_cosine(score: float) -> float:
    """Convert Atlas's cosine `vectorSearchScore` ((1 + cos) / 2, in 0..1) back to cosine."""
    return 2.0 * score - 1.0


def build_pipeline(
    query_vector: list[float], access_filter: dict[str, Any], k: int, index_name: str
) -> list[dict[str, Any]]:
    """Return the aggregation pipeline: pre-filtered `$vectorSearch`, then a projection with the score."""
    if not 1 <= k <= NUM_CANDIDATES:
        raise ValueError(f"k must be between 1 and {NUM_CANDIDATES}, got {k}")
    return [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": NUM_CANDIDATES,
                "limit": k,
                "filter": access_filter,
            }
        },
        {"$project": PROJECTION},
    ]


def to_chunk(doc: dict[str, Any]) -> RetrievedChunk:
    """Build a RetrievedChunk from one pipeline result, converting the score to cosine."""
    return RetrievedChunk(
        chunk_id=doc["_id"],
        doc_id=doc["doc_id"],
        doc_title=doc["doc_title"],
        doc_type=doc["doc_type"],
        page=doc["page"],
        section=doc["section"],
        text=doc["text"],
        score=atlas_score_to_cosine(doc["score"]),
        owner_stakeholder_id=doc.get("owner_stakeholder_id"),
    )


def apply_threshold(chunks: list[RetrievedChunk], min_score: float) -> list[RetrievedChunk]:
    """Keep chunks with cosine >= min_score (order preserved); [] means "not found"."""
    return [c for c in chunks if c.score >= min_score]


def bm25_text(doc: dict[str, Any]) -> str:
    """Text BM25 scores: same shape as the embedded text ("<title> > <section>\n<text>")."""
    return f"{doc['doc_title']} > {doc['section']}\n{doc['text']}"


def load_corpus(access_filter: dict[str, Any], company_id: str, role: Role,
                stakeholder_id: str | None) -> list[dict[str, Any]]:
    """Load every chunk this user may see, for the per-request BM25 index.

    Mongo applies `access_filter` (the same dict the vector search uses); then
    each document is re-checked with `is_visible`, and any disagreement raises
    PermissionError (fail closed) rather than silently dropping the chunk.
    """
    from app import db

    corpus = list(db.chunks().find(access_filter, CORPUS_PROJECTION).sort("_id", 1))
    leaked = [d["_id"] for d in corpus if not is_visible(d, company_id, role, stakeholder_id)]
    if leaked:
        raise PermissionError(f"access filter returned chunks the user may not see: {leaked}")
    return corpus


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product; embeddings are stored unit-norm, so this is the cosine."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def fuse(vector_hits: list[RetrievedChunk], bm25_ids: list[str], corpus: dict[str, dict[str, Any]],
         query_vector: list[float], k: int) -> list[RetrievedChunk]:
    """Merge the two rankings with RRF and return the top k, with both ranks recorded."""
    by_id = {c.chunk_id: c for c in vector_hits}
    vector_rank = {c.chunk_id: r for r, c in enumerate(vector_hits, start=1)}
    bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, start=1)}
    out: list[RetrievedChunk] = []
    for chunk_id, fused in rrf_fuse([[c.chunk_id for c in vector_hits], bm25_ids], k=RRF_K)[:k]:
        base = by_id.get(chunk_id)
        if base is None:  # BM25-only: build it from the corpus, cosine computed locally
            doc = corpus[chunk_id]
            base = to_chunk({**doc, "score": 0.0})
            base = replace(base, score=cosine(query_vector, doc["embedding"]))
        out.append(replace(base, vector_rank=vector_rank.get(chunk_id),
                           bm25_rank=bm25_rank.get(chunk_id), rrf_score=fused))
    return out


def retrieve(
    query: str,
    company_id: str,
    role: Role,
    stakeholder_id: str | None,
    k: int | None = None,
    min_score: float | None = None,
    mode: RetrievalMode | None = None,
) -> list[RetrievedChunk]:
    """Return up to k chunks this user may see, best first.

    k, min_score and mode default to `settings.retrieval_top_k`,
    `settings.retrieval_min_score` and `settings.retrieval_mode`.
    min_score applies to the vector cosine: vector hits below it are dropped,
    and if no vector hit passes, [] is returned at once (BM25 is not run), so
    the caller can answer "not found" without calling the LLM. In hybrid mode
    BM25 then adds keyword matches from the user's own corpus, merged by RRF.
    Raises (via build_access_filter) for an unknown role, a missing company, or
    an employee without a stakeholder id.
    """
    from app import db
    from app.config import settings
    from app.ingest.embedder import embed

    if not query.strip():
        raise ValueError("query is empty")
    access_filter = build_access_filter(company_id, role, stakeholder_id)  # fail closed before any work
    k = settings.retrieval_top_k if k is None else k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    mode = settings.retrieval_mode if mode is None else mode
    if mode not in ("vector", "hybrid"):
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    query_vector = embed([query])[0]
    depth = k if mode == "vector" else max(k, FUSION_DEPTH)
    pipeline = build_pipeline(query_vector, access_filter, depth, settings.vector_index_name)
    vector_hits = apply_threshold([to_chunk(doc) for doc in db.chunks().aggregate(pipeline)], min_score)
    if mode == "vector" or not vector_hits:
        return vector_hits[:k]

    corpus = load_corpus(access_filter, company_id, role, stakeholder_id)
    bm25_ids = bm25_ranking(query, [(d["_id"], bm25_text(d)) for d in corpus], FUSION_DEPTH)
    return fuse(vector_hits, bm25_ids, {d["_id"]: d for d in corpus}, query_vector, k)
