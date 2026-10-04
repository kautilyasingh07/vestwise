"""Document-level access: build_access_filter and grant-letter ownership at ingestion.

The filter tests need no DB or `.env`. Besides checking the filter's exact
shape (what Atlas receives), they evaluate it against sample chunks with a
tiny matcher, so they check *which documents* each role can see.
"""

from typing import Any

import pytest

from app.rag.access import build_access_filter

# One chunk per visibility case.
CHUNKS: dict[str, dict[str, Any]] = {
    "policy": {"company_id": "nimbus", "owner_stakeholder_id": None},
    "priya_letter": {"company_id": "nimbus", "owner_stakeholder_id": "sh_priya"},
    "rahul_letter": {"company_id": "nimbus", "owner_stakeholder_id": "sh_rahul"},
    "other_company_policy": {"company_id": "acme", "owner_stakeholder_id": None},
}


def matches(doc: dict[str, Any], flt: dict[str, Any]) -> bool:
    """Evaluate the subset of MQL the access filter uses: implicit AND, $or, $eq."""
    for key, cond in flt.items():
        if key == "$or":
            if not any(matches(doc, sub) for sub in cond):
                return False
        elif doc.get(key) != cond["$eq"]:
            return False
    return True


def visible(flt: dict[str, Any]) -> set[str]:
    """Names of the CHUNKS that pass the filter."""
    return {name for name, doc in CHUNKS.items() if matches(doc, flt)}


# --- admin ---


def test_admin_filter_is_company_only() -> None:
    assert build_access_filter("nimbus", "admin", "sh_arjun") == {"company_id": {"$eq": "nimbus"}}


def test_admin_sees_all_company_documents_and_nothing_else() -> None:
    assert visible(build_access_filter("nimbus", "admin", "sh_arjun")) == {"policy", "priya_letter", "rahul_letter"}


def test_admin_filter_does_not_depend_on_stakeholder_id() -> None:
    assert build_access_filter("nimbus", "admin", None) == build_access_filter("nimbus", "admin", "sh_arjun")


# --- employee ---


def test_employee_filter_shape() -> None:
    assert build_access_filter("nimbus", "employee", "sh_priya") == {
        "company_id": {"$eq": "nimbus"},
        "$or": [
            {"owner_stakeholder_id": {"$eq": None}},
            {"owner_stakeholder_id": {"$eq": "sh_priya"}},
        ],
    }


@pytest.mark.parametrize(("stakeholder_id", "expected"), [
    ("sh_priya", {"policy", "priya_letter"}),
    ("sh_rahul", {"policy", "rahul_letter"}),
    ("sh_kiran", {"policy"}),  # no letter of their own: company-wide documents only
])
def test_employee_sees_company_wide_plus_own_documents(stakeholder_id: str, expected: set[str]) -> None:
    assert visible(build_access_filter("nimbus", "employee", stakeholder_id)) == expected


def test_employee_filter_never_uses_null_inside_in() -> None:
    """Atlas rejects {"$in": [null, ...]} ("value type cannot be null"); the filter must use $or of $eq."""
    assert "$in" not in repr(build_access_filter("nimbus", "employee", "sh_priya"))


# --- fail closed ---


@pytest.mark.parametrize("stakeholder_id", [None, ""])
def test_employee_without_stakeholder_id_raises(stakeholder_id: str | None) -> None:
    with pytest.raises(ValueError, match="stakeholder_id"):
        build_access_filter("nimbus", "employee", stakeholder_id)


def test_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        build_access_filter("nimbus", "superuser", "sh_priya")  # type: ignore[arg-type]


def test_missing_company_raises() -> None:
    with pytest.raises(ValueError, match="company_id"):
        build_access_filter("", "admin", None)


# --- ingestion side: grant letters must have an owner ---


@pytest.fixture(scope="module")
def pipeline():  # noqa: ANN201 (module object)
    """app.ingest.pipeline imports config, which needs `.env`; skip if unavailable."""
    try:
        from app.ingest import pipeline as module
    except Exception as exc:  # missing .env -> pydantic ValidationError when config loads
        pytest.skip(f"config unavailable: {exc}")
    return module


def test_grant_letter_without_owner_is_refused(pipeline) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="owner_stakeholder_id"):
        pipeline.check_owner("grant_letter", None)


def test_company_wide_documents_need_no_owner(pipeline) -> None:  # noqa: ANN001
    pipeline.check_owner("policy", None)
    pipeline.check_owner("board_resolution", None)


def test_chunk_records_carry_owner(pipeline) -> None:  # noqa: ANN001
    from app.ingest.chunker import Chunk

    chunks = [Chunk(text="t", page=1, section="1. Key Terms", embed_text="x")]
    [record] = pipeline.chunk_records(chunks, [[0.0]], "nimbus_abc", "nimbus", "Grant Letter", "grant_letter", "sh_priya")
    assert record["owner_stakeholder_id"] == "sh_priya"
    [record] = pipeline.chunk_records(chunks, [[0.0]], "nimbus_abc", "nimbus", "ESOP Policy", "policy")
    assert record["owner_stakeholder_id"] is None
