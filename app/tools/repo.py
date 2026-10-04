"""Data access for the tools: the only module in app/tools that talks to MongoDB.

Every query is filtered by company_id (spec §7 multi-tenancy). Results come
back as plain dicts with dates as `datetime.date`, ready for the pure
functions in vesting.py and captable.py.
"""

from typing import Any

from app import db
from app.tools.vesting import to_date


class NotFoundError(LookupError):
    """Raised when a company the caller asked for does not exist."""


def _normalize_grant(grant: dict[str, Any]) -> dict[str, Any]:
    """Convert BSON datetimes to dates so the pure functions get one date type."""
    out = dict(grant)
    out["grant_date"] = to_date(out["grant_date"])
    if out.get("termination_date"):
        out["termination_date"] = to_date(out["termination_date"])
    return out


def get_pool_size(company_id: str) -> int:
    """ESOP pool size of a company."""
    company = db.companies().find_one({"_id": company_id}, {"esop_pool_size": 1})
    if company is None:
        raise NotFoundError(f"company {company_id!r} not found")
    return company["esop_pool_size"]


def get_holdings(company_id: str) -> list[dict[str, Any]]:
    """All issued-share holdings of a company."""
    return list(db.holdings().find({"company_id": company_id}))


def get_grants(company_id: str, stakeholder_id: str | None = None) -> list[dict[str, Any]]:
    """Grants of a company, optionally only one stakeholder's, oldest first."""
    query: dict[str, Any] = {"company_id": company_id}
    if stakeholder_id is not None:
        query["stakeholder_id"] = stakeholder_id
    return [_normalize_grant(g) for g in db.grants().find(query).sort("grant_date", 1)]


def get_stakeholder_names(company_id: str) -> dict[str, str]:
    """Map stakeholder_id -> name for a company (for display in cap table rows)."""
    return {s["_id"]: s["name"] for s in db.stakeholders().find({"company_id": company_id}, {"name": 1})}
