"""MCP server (spec FR-20): tools called through an in-memory MCP client. No Mongo, no LLM, no stdio.

The repository, retriever and audit store are faked. The tests check that each
MCP tool acts for the server's fixed user only, exactly as the agent's
build_tools tools do, and that every tool call is audit-logged, fail closed (FR-18).
"""

import asyncio
from typing import Any

import pytest

try:
    from langchain_core.tools import tool
    from mcp import Client
    from mcp.types import CallToolResult

    from app.audit import MAX_ARGS_CHARS, build_mcp_record
    from app.context import RequestContext, UnknownUserError
    from app.rag.retriever import RetrievedChunk
    from app.schemas import AuditRecord
    from app.tools import factory
    from mcp_server import server as mcp_mod
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)

PRIYA = RequestContext("u_priya", "employee", "nimbus", "sh_priya", "Priya Sharma")
ARJUN = RequestContext("u_arjun", "admin", "nimbus", "sh_arjun", "Arjun Rao")
IDENTITY_PARAMS = {"stakeholder_id", "company_id", "role", "user_id"}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[Any, ...]]]:
    """Fake the repo and the retriever; record what they were asked for."""
    seen: dict[str, list[tuple[Any, ...]]] = {"grants": [], "retrieve": []}

    def fake_grants(company_id: str, stakeholder_id: str) -> list:
        seen["grants"].append((company_id, stakeholder_id))
        return []

    def fake_retrieve(query: str, company_id: str, role: str, stakeholder_id: str | None) -> list:
        seen["retrieve"].append((query, company_id, role, stakeholder_id))
        return []

    monkeypatch.setattr(factory.repo, "get_grants", fake_grants)
    monkeypatch.setattr(factory, "retrieve", fake_retrieve)
    return seen


class FakeAudit:
    """Stands in for write_mcp_audit: keeps the records it would insert; `fail=True` raises."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.records: list[dict[str, Any]] = []

    def __call__(self, ctx: RequestContext, tool_name: str, arguments: dict[str, Any], latency_ms: int,
                 **kwargs: Any) -> str:
        if self.fail:
            raise ConnectionError("audit store down")
        self.records.append(build_mcp_record(ctx, tool_name, arguments, latency_ms, **kwargs))
        return f"audit_{len(self.records)}"


def list_tools(ctx: RequestContext) -> dict[str, Any]:
    return {t.name: t for t in asyncio.run(mcp_mod.build_server(ctx, FakeAudit()).list_tools())}


def client_call(ctx: RequestContext, name: str, args: dict[str, Any], audit: FakeAudit | None = None
                ) -> CallToolResult:
    """One tools/call through a real (in-memory) MCP client session, as Claude Desktop would send it."""

    async def go() -> CallToolResult:
        async with Client(mcp_mod.build_server(ctx, audit if audit is not None else FakeAudit())) as client:
            return await client.call_tool(name, args)

    return asyncio.run(go())


def call(ctx: RequestContext, name: str, args: dict[str, Any], audit: FakeAudit | None = None) -> str:
    result = client_call(ctx, name, args, audit)
    assert not result.is_error, result
    return result.content[0].text


# --- what is exposed ---

def test_employee_server_exposes_exactly_the_three_tools() -> None:
    assert set(list_tools(PRIYA)) == {"get_vesting_status", "get_grants", "search_policy"}


@pytest.mark.parametrize("ctx", [PRIYA, ARJUN], ids=["employee", "admin"])
def test_no_tool_takes_an_identity_argument(ctx: RequestContext) -> None:
    for name, t in list_tools(ctx).items():
        assert not IDENTITY_PARAMS & set(t.input_schema.get("properties", {})), name


def test_admin_server_has_stakeholder_name_but_no_cap_table() -> None:
    tools = list_tools(ARJUN)
    assert set(tools) == {"get_vesting_status", "get_grants", "search_policy"}
    assert "stakeholder_name" in tools["get_vesting_status"].input_schema["properties"]


def test_tools_are_marked_read_only() -> None:
    assert all(t.annotations.read_only_hint for t in list_tools(PRIYA).values())


def test_descriptions_come_from_build_tools() -> None:
    lc = {t.name: t.description for t in factory.build_tools(PRIYA)}
    assert {n: t.description for n, t in list_tools(PRIYA).items()} == {n: lc[n] for n in mcp_mod.EXPOSED}


# --- whose data each tool returns ---

def test_vesting_uses_the_server_user_and_the_given_date(calls: dict) -> None:
    out = call(PRIYA, "get_vesting_status", {"as_of": "2027-03-01"})
    assert calls["grants"] == [("nimbus", "sh_priya")]
    assert '"as_of": "2027-03-01"' in out and "Priya Sharma" in out


def test_identity_smuggled_in_arguments_is_ignored(calls: dict) -> None:
    out = call(PRIYA, "get_grants", {"stakeholder_id": "sh_rahul", "company_id": "other"})
    assert calls["grants"] == [("nimbus", "sh_priya")]
    assert "Rahul" not in out


def test_employee_cannot_name_another_stakeholder(calls: dict) -> None:
    out = call(PRIYA, "get_grants", {"stakeholder_name": "Rahul"})  # the parameter doesn't exist for employees
    assert calls["grants"] == [("nimbus", "sh_priya")]
    assert "Rahul" not in out


def test_search_policy_runs_with_the_users_access_context(calls: dict) -> None:
    out = call(PRIYA, "search_policy", {"query": "Rahul's grant letter"})
    assert calls["retrieve"] == [("Rahul's grant letter", "nimbus", "employee", "sh_priya")]
    assert out.startswith("NO_RESULTS")


def test_admin_tools_reach_other_stakeholders_by_name(calls: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "resolve_stakeholder", lambda company, name: ("sh_rahul", "Rahul Verma"))
    call(ARJUN, "get_grants", {"stakeholder_name": "Rahul"})
    assert calls["grants"] == [("nimbus", "sh_rahul")]


# --- startup: identity is fixed and must be valid ---

@pytest.mark.parametrize("user_id", [None, "", "   "])
def test_refuses_to_start_without_a_user(user_id: str | None) -> None:
    with pytest.raises(SystemExit, match="VESTWISE_USER_ID is not set"):
        mcp_mod.resolve_context(user_id)


def test_refuses_to_start_for_an_unknown_user(monkeypatch: pytest.MonkeyPatch) -> None:
    def unknown(user_id: str) -> RequestContext:
        raise UnknownUserError(f"unknown user {user_id!r}")

    monkeypatch.setattr(mcp_mod, "load_context", unknown)
    with pytest.raises(SystemExit, match="u_nobody"):
        mcp_mod.resolve_context("u_nobody")


def test_refuses_to_start_for_an_unusable_user_record(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(user_id: str) -> RequestContext:
        raise ValueError("employee 'u_x' has no stakeholder_id")

    monkeypatch.setattr(mcp_mod, "load_context", broken)
    with pytest.raises(SystemExit, match="no stakeholder_id"):
        mcp_mod.resolve_context("u_x")


def test_main_exits_before_serving_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_mod.settings, "vestwise_user_id", None)
    monkeypatch.setattr(mcp_mod, "build_server", lambda *a: pytest.fail("server built without a user"))
    with pytest.raises(SystemExit):
        mcp_mod.main()


def test_drift_between_build_tools_and_mcp_wrappers_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    @tool("get_grants")
    def changed(stakeholder_id: str) -> str:
        """A get_grants whose parameters no longer match the MCP wrapper."""
        return ""

    real = factory.build_tools
    monkeypatch.setattr(mcp_mod, "build_tools",
                        lambda ctx: [changed if t.name == "get_grants" else t for t in real(ctx)])
    with pytest.raises(RuntimeError, match="get_grants"):
        mcp_mod.build_server(PRIYA, FakeAudit())


# --- audit log (FR-18): one record per tool call, fail closed ---

CHUNK = RetrievedChunk("pol_017", "pol", "ESOP Policy", "policy", 7, "8. Acquisition and Corporate Actions",
                       "8.2 On a Change of Control, 50% ...", 0.49, None)


def test_every_call_writes_one_mcp_record(calls: dict) -> None:
    audit = FakeAudit()
    call(PRIYA, "get_vesting_status", {"as_of": "2027-03-01"}, audit)
    [record] = audit.records
    assert record["kind"] == "mcp" and record["outcome"] == "ok" and record["error"] is None
    assert (record["user_id"], record["role"], record["company_id"], record["stakeholder_id"]) == \
        ("u_priya", "employee", "nimbus", "sh_priya")
    assert record["tool_calls"] == [{"name": "get_vesting_status", "args": {"as_of": "2027-03-01"}}]
    assert isinstance(record["latency_ms"], int) and record["latency_ms"] >= 0
    assert record["model"] is None and record["answer"] is None  # our side runs no LLM; output not stored


def test_record_keeps_the_arguments_as_sent_including_ignored_ones(calls: dict) -> None:
    audit = FakeAudit()
    call(PRIYA, "get_grants", {"stakeholder_id": "sh_rahul"}, audit)
    assert audit.records[0]["tool_calls"][0]["args"] == {"stakeholder_id": "sh_rahul"}  # the attempt is visible
    assert calls["grants"] == [("nimbus", "sh_priya")]  # ...and still had no effect


def test_search_records_the_chunk_ids_it_returned(calls: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "retrieve", lambda *a, **k: [CHUNK])
    audit = FakeAudit()
    out = call(PRIYA, "search_policy", {"query": "change of control acquisition"}, audit)
    assert "[ESOP Policy, p. 7]" in out
    assert audit.records[0]["chunk_ids"] == ["pol_017"] and audit.records[0]["outcome"] == "ok"


def test_empty_search_is_not_found(calls: dict) -> None:
    audit = FakeAudit()
    call(PRIYA, "search_policy", {"query": "valuation"}, audit)
    assert audit.records[0]["outcome"] == "not_found" and audit.records[0]["chunk_ids"] == []


def test_tool_level_error_is_rejected(calls: dict) -> None:
    audit = FakeAudit()
    out = call(PRIYA, "get_vesting_status", {"as_of": "next March"}, audit)
    assert '"error"' in out
    assert audit.records[0]["outcome"] == "rejected" and audit.records[0]["error"]


@pytest.mark.parametrize(("name", "args"), [
    ("get_cap_table", {}),        # not exposed: unknown tool
    ("search_policy", {}),        # missing required argument
], ids=["unknown-tool", "invalid-arguments"])
def test_failed_calls_are_audited_too(calls: dict, name: str, args: dict) -> None:
    audit = FakeAudit()
    result = client_call(PRIYA, name, args, audit)
    assert result.is_error
    [record] = audit.records
    assert record["outcome"] == "error" and record["error"] and record["tool_calls"][0]["name"] == name


def test_tool_exception_is_audited_with_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*a: Any) -> list:
        raise ConnectionError("mongo unreachable")

    monkeypatch.setattr(factory.repo, "get_grants", down)
    audit = FakeAudit()
    result = client_call(PRIYA, "get_grants", {}, audit)
    assert result.is_error
    assert audit.records[0]["outcome"] == "error" and "mongo unreachable" in audit.records[0]["error"]


def test_audit_failure_withholds_the_result(calls: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory.repo, "get_grants", lambda *a: [{"_id": "g_priya_2025", "options": 4800}])
    result = client_call(PRIYA, "get_grants", {}, FakeAudit(fail=True))
    assert result.is_error
    text = result.content[0].text
    assert text == mcp_mod.AUDIT_FAILED
    assert "4800" not in text and "g_priya_2025" not in text


def test_audit_failure_after_a_tool_error_still_reports_the_tool_error(calls: dict) -> None:
    result = client_call(PRIYA, "get_cap_table", {}, FakeAudit(fail=True))
    assert result.is_error and "get_cap_table" in result.content[0].text


def test_mcp_record_is_listable_by_get_audit() -> None:
    record = build_mcp_record(PRIYA, "get_grants", {"stakeholder_id": "sh_rahul"}, 12, outcome="ok")
    parsed = AuditRecord(id="a1", **record)  # GET /audit's response model
    assert parsed.kind == "mcp" and parsed.model is None and parsed.tool_calls[0].name == "get_grants"


def test_oversized_arguments_are_truncated_in_the_record() -> None:
    record = build_mcp_record(PRIYA, "search_policy", {"query": "x" * 10_000}, 1, outcome="ok")
    args = record["tool_calls"][0]["args"]
    assert set(args) == {"_truncated"} and len(args["_truncated"]) == MAX_ARGS_CHARS


def test_main_wires_the_real_audit_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Stub:
        def run(self, transport: str) -> None:
            seen["transport"] = transport

    monkeypatch.setattr(mcp_mod.settings, "vestwise_user_id", "u_priya")
    monkeypatch.setattr(mcp_mod, "load_context", lambda uid: PRIYA)
    monkeypatch.setattr(mcp_mod, "build_server", lambda ctx, audit: seen.update(audit=audit) or Stub())
    mcp_mod.main()
    assert seen == {"audit": mcp_mod.write_mcp_audit, "transport": "stdio"}
