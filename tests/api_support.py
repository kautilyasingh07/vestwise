"""In-memory fakes for API tests: users, repo, agent, audit store, ingester. No Mongo, no LLM.

Data comes from data/seed.json, so the API returns the real seed numbers
(Priya 2,100 vested on 2026-10-03, Arjun 57.14% fully diluted) without a database.
Import only after app config is available (tests skip otherwise).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app import main
from app.audit import build_record
from app.context import RequestContext, UnknownUserError
from app.ingest.pipeline import IngestResult

SEED = json.loads((Path(__file__).resolve().parent.parent / "data" / "seed.json").read_text(encoding="utf-8"))
OTHER_COMPANY_USER = RequestContext("u_other", "admin", "acme", "sh_acme_admin", "Other Admin")
BROKEN_USER_ID = "u_broken"  # exists, but its record is unusable -> 403


def contexts() -> dict[str, RequestContext]:
    """RequestContext for every seeded user, plus an admin of another company."""
    out = {u["_id"]: RequestContext(u["_id"], u["role"], u["company_id"], u["stakeholder_id"], u["name"])
           for u in SEED["users"]}
    out[OTHER_COMPANY_USER.user_id] = OTHER_COMPANY_USER
    return out


def fake_loader(user_id: str) -> RequestContext:
    """Stands in for load_context: same exceptions, no Mongo."""
    if user_id == BROKEN_USER_ID:
        raise ValueError("employee has no stakeholder_id")
    try:
        return contexts()[user_id]
    except KeyError:
        raise UnknownUserError(user_id) from None


class FakeRepo:
    """The four repo functions the API uses, over seed.json, scoped by company like the real one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _company(self, company_id: str) -> str:
        return SEED["company"]["_id"] if company_id == SEED["company"]["_id"] else ""

    def get_grants(self, company_id: str, stakeholder_id: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("get_grants", (company_id, stakeholder_id)))
        rows = [g for g in SEED["grants"] if g["company_id"] == company_id
                and (stakeholder_id is None or g["stakeholder_id"] == stakeholder_id)]
        return [{**g, "grant_date": date.fromisoformat(g["grant_date"]),
                 "termination_date": date.fromisoformat(g["termination_date"]) if g["termination_date"] else None}
                for g in rows]

    def get_holdings(self, company_id: str) -> list[dict[str, Any]]:
        self.calls.append(("get_holdings", (company_id,)))
        return [h for h in SEED["holdings"] if h["company_id"] == company_id]

    def get_pool_size(self, company_id: str) -> int:
        self.calls.append(("get_pool_size", (company_id,)))
        return SEED["company"]["esop_pool_size"]

    def get_stakeholder_names(self, company_id: str) -> dict[str, str]:
        self.calls.append(("get_stakeholder_names", (company_id,)))
        return {s["_id"]: s["name"] for s in SEED["stakeholders"] if s["company_id"] == company_id}


@dataclass
class FakeAgent:
    """Records every call; returns `result` or raises `error`."""

    result: dict[str, Any] = field(default_factory=lambda: {
        "answer": "You have vested 2,100 options [ESOP Policy, p. 3].",
        "citations": [{"doc_title": "ESOP Policy", "page": 3, "section": "4. Vesting",
                       "snippet": "4.1 Vesting period ...", "chunk_id": "nimbus_x_007"}],
        "tool_calls": [{"name": "get_vesting_status", "args": {"as_of": "2026-10-03"}}],
        "chunk_ids": ["nimbus_x_007", "nimbus_x_008"],
        "flags": [],
        "citation_check": {"total": 1, "invalid": 0, "retried": False, "stripped": 0},
    })
    error: Exception | None = None
    calls: list[tuple[RequestContext, str, list[dict[str, str]], date | None]] = field(default_factory=list)

    def __call__(self, ctx: RequestContext, message: str, history: list[dict[str, str]],
                 as_of: date | None = None) -> dict[str, Any]:
        self.calls.append((ctx, message, history, as_of))
        if self.error is not None:
            raise self.error
        return self.result


@dataclass
class FakeAudit:
    """In-memory audit_logs using the real build_record, so record shape is the real one."""

    records: list[dict[str, Any]] = field(default_factory=list)
    fail: bool = False

    def write(self, ctx: RequestContext, question: str, chunk_ids: list[str], tool_calls: list[dict[str, Any]],
              answer: str | None, latency_ms: int, *, as_of: date | None = None, error: str | None = None,
              flags: list[str] | None = None, citation_check: dict[str, Any] | None = None) -> str:
        if self.fail:
            raise ConnectionError("audit store down")
        record = build_record(ctx, question, chunk_ids, tool_calls, answer, latency_ms, as_of=as_of, error=error,
                              flags=flags, citation_check=citation_check)
        record["id"] = f"audit_{len(self.records) + 1}"
        self.records.append(record)
        return record["id"]

    def read(self, company_id: str, limit: int) -> list[dict[str, Any]]:
        mine = [r for r in self.records if r["company_id"] == company_id]
        return list(reversed(mine))[:limit]


@dataclass
class FakeIngester:
    """Records ingest_file calls and returns a plausible IngestResult (or raises `error`)."""

    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, path: Path, company_id: str, doc_type: str, title: str,
                 owner_stakeholder_id: str | None = None) -> IngestResult:
        self.calls.append({"filename": path.name, "bytes": path.read_bytes(), "company_id": company_id,
                           "doc_type": doc_type, "title": title, "owner": owner_stakeholder_id})
        if self.error is not None:
            raise self.error
        return IngestResult(doc_id=f"{company_id}_abc123", title=title, owner_stakeholder_id=owner_stakeholder_id,
                            pages=2, chunks=8, replaced_chunks=0)


@dataclass
class Api:
    """A TestClient wired to fakes, plus handles to inspect them."""

    client: TestClient
    agent: FakeAgent
    audit: FakeAudit
    repo: FakeRepo
    ingester: FakeIngester

    def as_user(self, user_id: str | None) -> dict[str, str]:
        """Headers for a request as this user (no header if None)."""
        return {} if user_id is None else {"X-User-Id": user_id}


def make_api() -> Iterator[Api]:
    """Install fakes via dependency_overrides, yield the Api, then remove the overrides."""
    api = Api(TestClient(main.app), FakeAgent(), FakeAudit(), FakeRepo(), FakeIngester())
    main.app.dependency_overrides.update({
        main.get_user_loader: lambda: fake_loader,
        main.get_agent: lambda: api.agent,
        main.get_audit_writer: lambda: api.audit.write,
        main.get_audit_reader: lambda: api.audit.read,
        main.get_repo: lambda: api.repo,
        main.get_ingester: lambda: api.ingester,
    })
    try:
        yield api
    finally:
        main.app.dependency_overrides.clear()


def forbid_network(monkeypatch: Any) -> None:
    """Make any Mongo or LLM use fail loudly, proving a test runs entirely on fakes."""
    from app import agent, db

    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("network access in a test that should use fakes")

    monkeypatch.setattr(db, "get_client", boom)
    monkeypatch.setattr(agent, "get_chat_model", boom)
