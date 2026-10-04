"""Tests for the pure parts of app/rag/retriever.py (spec §8.2). No Mongo, model or `.env` needed."""

import pytest

from app.rag.access import build_access_filter
from app.rag.retriever import (
    NUM_CANDIDATES,
    PROJECTION,
    RetrievedChunk,
    apply_threshold,
    atlas_score_to_cosine,
    build_pipeline,
    to_chunk,
)

VECTOR = [0.0] * 383 + [1.0]


def make_chunk(score: float, chunk_id: str = "c") -> RetrievedChunk:
    """A RetrievedChunk with only the score varying."""
    return RetrievedChunk(chunk_id, "d", "ESOP Policy", "policy", 5, "6. Exercise Window", "text", score, None)


def test_pipeline_uses_access_filter_candidates_and_limit() -> None:
    access = build_access_filter("nimbus", "employee", "sh_priya")
    search = build_pipeline(VECTOR, access, 5, "idx")[0]["$vectorSearch"]
    assert search["filter"] == access
    assert search["numCandidates"] == NUM_CANDIDATES == 100
    assert search["limit"] == 5
    assert search["index"] == "idx" and search["path"] == "embedding"


def test_pipeline_filter_is_inside_vector_search_not_a_later_match() -> None:
    pipeline = build_pipeline(VECTOR, build_access_filter("nimbus", "admin", None), 5, "idx")
    assert [next(iter(stage)) for stage in pipeline] == ["$vectorSearch", "$project"]


def test_projection_returns_score_and_never_the_embedding() -> None:
    assert PROJECTION["score"] == {"$meta": "vectorSearchScore"}
    assert "embedding" not in PROJECTION
    assert {"doc_title", "page", "section", "text", "owner_stakeholder_id"} <= PROJECTION.keys()


@pytest.mark.parametrize("k", [0, -1, NUM_CANDIDATES + 1])
def test_pipeline_rejects_k_outside_candidate_pool(k: int) -> None:
    with pytest.raises(ValueError):
        build_pipeline(VECTOR, {"company_id": {"$eq": "nimbus"}}, k, "idx")


@pytest.mark.parametrize(("atlas", "cosine"), [(1.0, 1.0), (0.5, 0.0), (0.0, -1.0), (0.8124, 0.6248)])
def test_atlas_score_to_cosine(atlas: float, cosine: float) -> None:
    # Atlas cosine score = (1 + cos) / 2; last row was measured on our index (Phase 4 gotcha).
    assert atlas_score_to_cosine(atlas) == pytest.approx(cosine)


def test_to_chunk_converts_score_and_maps_fields() -> None:
    doc = {"_id": "nimbus_x_012", "doc_id": "nimbus_x", "doc_title": "ESOP Policy", "doc_type": "policy",
           "page": 5, "section": "6. Exercise Window After Leaving", "text": "6.1 ...", "score": 0.71,
           "owner_stakeholder_id": None}
    chunk = to_chunk(doc)
    assert chunk.chunk_id == "nimbus_x_012"
    assert chunk.page == 5
    assert chunk.score == pytest.approx(0.42)
    assert chunk.owner_stakeholder_id is None


def test_threshold_keeps_order_and_boundary() -> None:
    chunks = [make_chunk(0.6, "a"), make_chunk(0.35, "b"), make_chunk(0.2, "c")]
    assert [c.chunk_id for c in apply_threshold(chunks, 0.35)] == ["a", "b"]


def test_threshold_returns_empty_when_nothing_passes() -> None:
    assert apply_threshold([make_chunk(0.1), make_chunk(0.2)], 0.35) == []
