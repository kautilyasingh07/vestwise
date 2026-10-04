"""Policy retrieval: `$vectorSearch` with the access pre-filter and a score threshold (spec §8.2).

All of `company_id`, `role` and `stakeholder_id` come from the server-side
request context, never from the LLM. The access filter is applied inside the
vector search, so forbidden chunks are never candidates.

Scores are returned as **cosine similarity**. Atlas reports
`vectorSearchScore = (1 + cosine) / 2` for a cosine index, so a raw Atlas score
of 0.71 is a cosine of 0.42; we convert back before thresholding, otherwise the
spec's cosine threshold would be compared against the wrong scale.

Mongo, config and the embedding model are imported lazily inside `retrieve`,
so the pure helpers here can be tested without `.env` (same pattern as the chunker).
"""

from dataclasses import dataclass
from typing import Any

from app.rag.access import Role, build_access_filter

NUM_CANDIDATES = 100  # spec §8.2: HNSW candidate pool; `limit` (k) is taken from it

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


def retrieve(
    query: str,
    company_id: str,
    role: Role,
    stakeholder_id: str | None,
    k: int | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Return up to k chunks this user may see, best first, each with cosine >= min_score.

    k and min_score default to `settings.retrieval_top_k` / `settings.retrieval_min_score`.
    Returns [] when nothing passes the threshold, so the caller can answer
    "not found" without calling the LLM. Raises (via build_access_filter) for an
    unknown role, a missing company, or an employee without a stakeholder id.
    """
    from app import db
    from app.config import settings
    from app.ingest.embedder import embed

    if not query.strip():
        raise ValueError("query is empty")
    access_filter = build_access_filter(company_id, role, stakeholder_id)  # fail closed before any work
    k = settings.retrieval_top_k if k is None else k
    min_score = settings.retrieval_min_score if min_score is None else min_score

    query_vector = embed([query])[0]
    pipeline = build_pipeline(query_vector, access_filter, k, settings.vector_index_name)
    chunks = [to_chunk(doc) for doc in db.chunks().aggregate(pipeline)]
    return apply_threshold(chunks, min_score)
