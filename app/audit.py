"""Audit log: one record per /chat request (spec FR-18, §7 `audit_logs`, §5 Traceability).

A record must let someone answer "why did the bot say this?" later, without
re-running anything: who asked, as which role, for which date, which model
answered, which chunks it saw, which tools it called with which arguments,
what it said, how long it took, and whether it failed.
"""

from datetime import UTC, date, datetime
from typing import Any, Literal

from app import db
from app.config import settings
from app.context import RequestContext
from app.rag.prompts import ACCESS_DENIED_MESSAGE, NOT_FOUND_MESSAGE

Outcome = Literal["answered", "not_found", "refused", "error"]

MAX_LIMIT = 200


def classify(answer: str | None, error: str | None) -> Outcome:
    """Coarse outcome of a turn. The one classifier for both the /chat response's `outcome`
    (the UI styles by it) and the audit record, so the two can never disagree."""
    if error is not None:
        return "error"
    if answer == NOT_FOUND_MESSAGE:
        return "not_found"
    if answer == ACCESS_DENIED_MESSAGE:
        return "refused"
    return "answered"


def build_record(
    ctx: RequestContext,
    question: str,
    chunk_ids: list[str],
    tool_calls: list[dict[str, Any]],
    answer: str | None,
    latency_ms: int,
    *,
    as_of: date | None = None,
    error: str | None = None,
    flags: list[str] | None = None,
    citation_check: dict[str, Any] | None = None,
    ts: datetime | None = None,
) -> dict[str, Any]:
    """The `audit_logs` document. Pure: no I/O, so it is unit-testable.

    `flags` carries quality signals from the agent, e.g. "citation_invalid" when an
    in-text citation pointed at a page that wasn't retrieved and had to be stripped.
    """
    return {
        "company_id": ctx.company_id,
        "user_id": ctx.user_id,
        "role": ctx.role,
        "stakeholder_id": ctx.stakeholder_id,
        "question": question,
        "as_of": as_of.isoformat() if as_of else None,  # BSON has no date-only type
        "model": f"{settings.llm_provider}/{settings.llm_model}",
        "chunk_ids": chunk_ids,
        "tool_calls": tool_calls,
        "answer": answer,
        "outcome": classify(answer, error),
        "error": error,
        "flags": flags or [],
        "citation_check": citation_check,
        "latency_ms": latency_ms,
        "ts": ts or datetime.now(UTC),
    }


def write_audit(
    ctx: RequestContext,
    question: str,
    chunk_ids: list[str],
    tool_calls: list[dict[str, Any]],
    answer: str | None,
    latency_ms: int,
    *,
    as_of: date | None = None,
    error: str | None = None,
    flags: list[str] | None = None,
    citation_check: dict[str, Any] | None = None,
) -> str:
    """Insert one audit record and return its id.

    Example: write_audit(ctx, "How many options have I vested?", [], [{"name": "get_vesting_status",
    "args": {"as_of": "2026-10-03"}}], "You have vested 2,100 options.", 1840) -> "6702f0..."
    """
    record = build_record(ctx, question, chunk_ids, tool_calls, answer, latency_ms, as_of=as_of, error=error,
                          flags=flags, citation_check=citation_check)
    return str(db.audit_logs().insert_one(record).inserted_id)


ComplianceOutcome = Literal["compliant", "issues_found", "error"]


def build_compliance_record(
    ctx: RequestContext,
    file_name: str,
    latency_ms: int,
    *,
    outcome: ComplianceOutcome,
    summary: str | None = None,
    details: dict[str, Any] | None = None,
    flags: list[str] | None = None,
    error: str | None = None,
    ts: datetime | None = None,
) -> dict[str, Any]:
    """The audit_logs document for one POST /compliance/check (FR-25). Pure.

    Same top-level shape as a chat record (so GET /audit lists both), with `kind`
    "compliance_check" and the check itself under `compliance`: file hash, letter
    title, counts, and each finding's id, field, status and rule ids.
    """
    return {
        "kind": "compliance_check",
        "company_id": ctx.company_id,
        "user_id": ctx.user_id,
        "role": ctx.role,
        "stakeholder_id": ctx.stakeholder_id,
        "question": f"Compliance check: {file_name}",
        "as_of": None,
        "model": f"{settings.llm_provider}/{settings.llm_model}",
        "chunk_ids": [],
        "tool_calls": [],
        "answer": summary,
        "outcome": outcome,
        "error": error,
        "flags": flags or [],
        "citation_check": None,
        "compliance": details,
        "latency_ms": latency_ms,
        "ts": ts or datetime.now(UTC),
    }


def write_compliance_audit(ctx: RequestContext, file_name: str, latency_ms: int, **kwargs: Any) -> str:
    """Insert one compliance-check audit record (see build_compliance_record) and return its id."""
    record = build_compliance_record(ctx, file_name, latency_ms, **kwargs)
    return str(db.audit_logs().insert_one(record).inserted_id)


def read_audit(company_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Latest audit records of one company, newest first (uses the company_id + ts index).

    Mongo's ObjectId `_id` is returned as a string `id`. BSON stores UTC, but pymongo
    returns naive datetimes by default, so `ts` is marked UTC here (else it serialises
    without an offset and reads like local time).
    """
    limit = max(1, min(limit, MAX_LIMIT))
    cursor = db.audit_logs().find({"company_id": company_id}).sort("ts", -1).limit(limit)
    return [{"id": str(doc.pop("_id")), **doc, "ts": as_utc(doc["ts"])} for doc in cursor]


def as_utc(ts: datetime) -> datetime:
    """Attach UTC to a naive datetime read from Mongo; leave aware ones alone."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
