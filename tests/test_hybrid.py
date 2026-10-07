"""Hybrid retrieval (spec §8.2 "Should"): BM25 tokenising/ranking, RRF, and BM25 access control.

The pure tests need nothing. The retrieve() tests fake Mongo and the embedder
(no network) but import app.config, so they skip without `.env`, like test_access.py.
"""

from typing import Any

import pytest

from app.rag.access import build_access_filter, is_visible
from app.rag.hybrid import RRF_K, bm25_ranking, rrf_fuse, stem, tokenize

# --- tokenizer and stemmer ---


@pytest.mark.parametrize("word", ["acquired", "acquires", "acquiring", "acquire"])
def test_inflections_share_a_stem(word: str) -> None:
    assert stem(word) == "acquir"


def test_tokenize_keeps_clause_numbers_and_drops_stopwords() -> None:
    assert tokenize("What does clause 8.2 say about Change of Control?") == ["claus", "8.2", "say", "chang", "control"]


def test_stem_leaves_double_s_and_short_words() -> None:
    assert stem("process") == "process"
    assert stem("has") == "has"


# --- BM25 ---

DOCS = [
    ("policy_8", "ESOP Policy > 8. Acquisition\n8.2 On a Change of Control, 50% of unvested options vest."),
    ("policy_6", "ESOP Policy > 6. Exercise Window\n6.1 You may exercise vested options within 90 days."),
    ("policy_9", "ESOP Policy > 9. Tax\n9.1 Tax on exercise is a perquisite."),
]


def test_bm25_ranks_the_keyword_match_first() -> None:
    assert bm25_ranking("change of control", DOCS, 3)[0] == "policy_8"
    assert bm25_ranking("clause 8.2", DOCS, 3) == ["policy_8"]


def test_bm25_returns_nothing_without_term_overlap() -> None:
    assert bm25_ranking("weather in Bengaluru", DOCS, 3) == []
    assert bm25_ranking("the of and", DOCS, 3) == []  # only stopwords
    assert bm25_ranking("change of control", [], 3) == []


def test_bm25_respects_n() -> None:
    assert len(bm25_ranking("options", DOCS, 1)) == 1


# --- reciprocal rank fusion ---


def test_rrf_worked_example() -> None:
    # vector: a, b, c   bm25: c, a, d   (k = 60)
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "a", "d"]]))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in rrf_fuse([["a", "b", "c"], ["c", "a", "d"]])] == ["a", "c", "b", "d"]


def test_rrf_default_k_is_60_and_ties_go_to_first_ranking() -> None:
    assert RRF_K == 60
    assert [d for d, _ in rrf_fuse([["x"], ["y"]])] == ["x", "y"]


# --- is_visible mirrors build_access_filter ---

POLICY = {"company_id": "nimbus", "owner_stakeholder_id": None}
PRIYA_LETTER = {"company_id": "nimbus", "owner_stakeholder_id": "sh_priya"}
RAHUL_LETTER = {"company_id": "nimbus", "owner_stakeholder_id": "sh_rahul"}
OTHER_CO = {"company_id": "acme", "owner_stakeholder_id": None}


@pytest.mark.parametrize(("chunk", "employee", "admin"), [
    (POLICY, True, True), (PRIYA_LETTER, True, True), (RAHUL_LETTER, False, True), (OTHER_CO, False, False),
])
def test_is_visible(chunk: dict, employee: bool, admin: bool) -> None:
    assert is_visible(chunk, "nimbus", "employee", "sh_priya") is employee
    assert is_visible(chunk, "nimbus", "admin", None) is admin


def test_is_visible_fails_closed_like_the_filter() -> None:
    with pytest.raises(ValueError):
        is_visible(POLICY, "nimbus", "employee", None)
    with pytest.raises(ValueError):
        is_visible(POLICY, "nimbus", "superuser", "sh_priya")  # type: ignore[arg-type]


# --- retrieve() in hybrid mode, Mongo and embedder faked ---

UNIT = [1.0] + [0.0] * 383


def chunk_doc(cid: str, owner: str | None, text: str, title: str = "ESOP Policy") -> dict[str, Any]:
    return {"_id": cid, "company_id": "nimbus", "owner_stakeholder_id": owner, "doc_id": cid.rsplit("_", 1)[0],
            "doc_title": title, "doc_type": "grant_letter" if owner else "policy", "page": 1,
            "section": "1. Terms", "text": text, "embedding": UNIT}


# Rahul's letter holds a word nobody else's chunks have, so BM25 would rank it
# first for "zebracorn" if it were in the corpus at all.
STORE = [
    chunk_doc("policy_001", None, "Options vest monthly after the cliff."),
    chunk_doc("priya_001", "sh_priya", "Priya Sharma is granted 3,600 options.", "Grant Letter: Priya Sharma"),
    chunk_doc("rahul_001", "sh_rahul", "Rahul Verma zebracorn clause: 2,400 options.", "Grant Letter: Rahul Verma"),
]


def matches(doc: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Tiny evaluator for the filter shapes build_access_filter produces ($eq, $or)."""
    for key, cond in flt.items():
        if key == "$or":
            if not any(matches(doc, sub) for sub in cond):
                return False
        elif doc.get(key) != cond["$eq"]:
            return False
    return True


class Cursor(list):
    def sort(self, key: str, direction: int) -> "Cursor":
        return Cursor(sorted(self, key=lambda d: d[key], reverse=direction < 0))


class FakeChunks:
    """Applies the access filter like Mongo would; `leaky=True` ignores it on find() (a bug)."""

    def __init__(self, vector_score: float = 0.9, leaky: bool = False) -> None:
        self.vector_score, self.leaky = vector_score, leaky
        self.find_filters: list[dict[str, Any]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flt = pipeline[0]["$vectorSearch"]["filter"]
        # Vector search only ever surfaces the policy chunk; Atlas score = (1 + cos) / 2.
        return [{**d, "score": (1 + self.vector_score) / 2} for d in STORE
                if d["_id"] == "policy_001" and matches(d, flt)]

    def find(self, flt: dict[str, Any], projection: dict[str, Any]) -> Cursor:
        self.find_filters.append(flt)
        return Cursor(dict(d) for d in STORE if self.leaky or matches(d, flt))


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> Any:
    try:
        from app import db
        from app.ingest import embedder
        from app.rag import retriever
    except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
        pytest.skip(f"config unavailable: {exc}")

    def install(**kwargs: Any) -> FakeChunks:
        store = FakeChunks(**kwargs)
        monkeypatch.setattr(db, "chunks", lambda: store)
        monkeypatch.setattr(embedder, "embed", lambda texts: [UNIT for _ in texts])
        return store

    return retriever, install


def test_priya_cannot_get_rahuls_chunk_through_bm25(fake_store: Any) -> None:
    retriever, install = fake_store
    store = install()
    hits = retriever.retrieve("zebracorn Rahul Verma", "nimbus", "employee", "sh_priya", k=5, mode="hybrid")
    assert hits, "gate should pass (policy chunk cosine 0.9)"
    assert all(h.owner_stakeholder_id != "sh_rahul" for h in hits)
    # The BM25 corpus was loaded with exactly the vector search's access filter.
    assert store.find_filters == [build_access_filter("nimbus", "employee", "sh_priya")]


def test_admin_positive_control_gets_rahuls_chunk_via_bm25_only(fake_store: Any) -> None:
    retriever, install = fake_store
    install()
    hits = retriever.retrieve("zebracorn", "nimbus", "admin", None, k=5, mode="hybrid")
    rahul = next(h for h in hits if h.chunk_id == "rahul_001")
    assert rahul.vector_rank is None and rahul.bm25_rank == 1
    assert rahul.score == pytest.approx(1.0)  # cosine computed locally from the stored embedding


def test_a_leaky_corpus_query_fails_closed(fake_store: Any) -> None:
    retriever, install = fake_store
    install(leaky=True)
    with pytest.raises(PermissionError, match="rahul_001"):
        retriever.retrieve("zebracorn", "nimbus", "employee", "sh_priya", k=5, mode="hybrid")


def test_not_found_shortcut_skips_bm25(fake_store: Any) -> None:
    retriever, install = fake_store
    store = install(vector_score=0.10)
    assert retriever.retrieve("zebracorn", "nimbus", "admin", None, k=5, min_score=0.35, mode="hybrid") == []
    assert store.find_filters == []  # BM25 corpus never loaded


def test_vector_mode_never_loads_the_bm25_corpus(fake_store: Any) -> None:
    retriever, install = fake_store
    store = install()
    hits = retriever.retrieve("zebracorn", "nimbus", "admin", None, k=5, mode="vector")
    assert [h.chunk_id for h in hits] == ["policy_001"] and hits[0].rrf_score is None
    assert store.find_filters == []


def test_unknown_mode_is_rejected(fake_store: Any) -> None:
    retriever, install = fake_store
    install()
    with pytest.raises(ValueError, match="mode"):
        retriever.retrieve("x", "nimbus", "admin", None, mode="keyword")  # type: ignore[arg-type]
