"""Document-level access filter for `$vectorSearch` (spec §8.2, FR-6, FR-17).

Chunks carry `company_id` and `owner_stakeholder_id` (None = company-wide,
e.g. the policy; a stakeholder id = personal, e.g. a grant letter). The filter
is built from the server-side request context, never from the LLM or the
request body, and is applied inside the vector search (both fields are
`filter` fields in the Atlas index), so a forbidden chunk is never a candidate.
"""

from typing import Any, Literal

Role = Literal["employee", "admin"]


def build_access_filter(company_id: str, role: Role, stakeholder_id: str | None) -> dict[str, Any]:
    """Return the `$vectorSearch` filter for this user.

    admin    -> every document in their company.
    employee -> company-wide documents plus documents they own.
    Fails closed: an unknown role, a missing company, or an employee without a
    stakeholder id raises instead of returning a broader filter.
    """
    if not company_id:
        raise ValueError("company_id is required")
    if role == "admin":
        return {"company_id": {"$eq": company_id}}
    if role == "employee":
        if not stakeholder_id:
            raise ValueError("employee access filter needs the employee's stakeholder_id")
        # $or of two $eq, not {"$in": [None, id]}: Atlas rejects null inside $in.
        return {
            "company_id": {"$eq": company_id},
            "$or": [
                {"owner_stakeholder_id": {"$eq": None}},
                {"owner_stakeholder_id": {"$eq": stakeholder_id}},
            ],
        }
    raise ValueError(f"unknown role: {role!r}")


def is_visible(chunk: dict[str, Any], company_id: str, role: Role, stakeholder_id: str | None) -> bool:
    """Python mirror of build_access_filter, for checking chunks already loaded into memory.

    Used as a fail-closed second check on the BM25 corpus (which Mongo filtered
    with build_access_filter): if this ever disagrees with Mongo, retrieval raises.
    """
    build_access_filter(company_id, role, stakeholder_id)  # same argument validation, same errors
    if chunk.get("company_id") != company_id:
        return False
    if role == "admin":
        return True
    return chunk.get("owner_stakeholder_id") in (None, stakeholder_id)
