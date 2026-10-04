"""Tool factory and system prompt tests (spec §8.6, FR-17, §12 test_access cases). No LLM, no Mongo.

The repo and retriever are monkeypatched, so these check *which identity* each
tool uses, not the data. Importing the factory loads config, which needs `.env`.
"""

from datetime import date
from typing import Any

import pytest

try:
    from app.context import RequestContext
    from app.rag import prompts
    from app.rag.retriever import RetrievedChunk
    from app.tools import factory
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)

PRIYA = RequestContext("u_priya", "employee", "nimbus", "sh_priya", "Priya Sharma")
ARJUN = RequestContext("u_arjun", "admin", "nimbus", "sh_arjun", "Arjun Mehta")
FORBIDDEN_PARAMS = {"company_id", "role", "stakeholder_id", "user_id"}
AS_OF = date(2026, 10, 3)

PRIYA_GRANT = {
    "_id": "g_priya_2025", "company_id": "nimbus", "stakeholder_id": "sh_priya",
    "grant_date": date(2025, 1, 1), "options": 4800, "strike_price": 10,
    "vesting": {"cliff_months": 12, "total_months": 48, "frequency": "monthly"},
    "exercised": 0, "lapsed": 0, "status": "active", "termination_date": None,
}
NAMES = {"sh_priya": "Priya Sharma", "sh_rahul": "Rahul Verma", "sh_kiran": "Kiran Rao",
         "sh_arjun": "Arjun Mehta", "sh_ravi": "Ravi Kumar", "sh_ravi2": "Ravi Shah"}


def tools_by_name(ctx: RequestContext) -> dict[str, Any]:
    return {t.name: t for t in factory.build_tools(ctx, AS_OF)}


@pytest.fixture
def grant_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """Record (company_id, stakeholder_id) for every repo.get_grants call; return Priya's grant for her."""
    calls: list[tuple[str, str | None]] = []

    def fake_get_grants(company_id: str, stakeholder_id: str | None = None) -> list[dict[str, Any]]:
        calls.append((company_id, stakeholder_id))
        return [PRIYA_GRANT] if stakeholder_id == "sh_priya" else []

    monkeypatch.setattr(factory.repo, "get_grants", fake_get_grants)
    monkeypatch.setattr(factory.repo, "get_stakeholder_names", lambda company_id: NAMES)
    return calls


# --- which tools exist, and with which parameters ---

def test_employee_gets_exactly_three_tools() -> None:
    assert set(tools_by_name(PRIYA)) == {"search_policy", "get_vesting_status", "get_grants"}


def test_employee_has_no_admin_tools() -> None:
    assert not {"get_cap_table", "simulate_dilution"} & set(tools_by_name(PRIYA))


def test_admin_gets_five_tools() -> None:
    assert set(tools_by_name(ARJUN)) == {"search_policy", "get_vesting_status", "get_grants",
                                         "get_cap_table", "simulate_dilution"}


@pytest.mark.parametrize("ctx", [PRIYA, ARJUN], ids=["employee", "admin"])
def test_no_tool_accepts_identity_parameters(ctx: RequestContext) -> None:
    for tool in factory.build_tools(ctx):
        assert not FORBIDDEN_PARAMS & set(tool.args), f"{tool.name} exposes {set(tool.args)}"


def test_employee_tool_parameters() -> None:
    tools = tools_by_name(PRIYA)
    assert set(tools["search_policy"].args) == {"query"}
    assert set(tools["get_vesting_status"].args) == {"as_of"}
    assert set(tools["get_grants"].args) == set()


def test_admin_grant_tools_take_stakeholder_name() -> None:
    tools = tools_by_name(ARJUN)
    assert set(tools["get_vesting_status"].args) == {"as_of", "stakeholder_name"}
    assert set(tools["simulate_dilution"].args) == {"new_shares", "investor_name"}


@pytest.mark.parametrize("ctx", [PRIYA, ARJUN], ids=["employee", "admin"])
def test_every_tool_has_a_docstring_with_an_example(ctx: RequestContext) -> None:
    for tool in factory.build_tools(ctx):
        assert "Example" in tool.description, tool.name


# --- closures: identity comes from ctx, whatever the model sends ---

def test_search_policy_uses_ctx_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_retrieve(query: str, company_id: str, role: str, stakeholder_id: str | None) -> list:
        seen.update(query=query, company_id=company_id, role=role, stakeholder_id=stakeholder_id)
        return []

    monkeypatch.setattr(factory, "retrieve", fake_retrieve)
    # The model tries to smuggle identity in; the schema has no such fields, so they're dropped.
    tools_by_name(PRIYA)["search_policy"].invoke(
        {"query": "vesting", "stakeholder_id": "sh_rahul", "role": "admin", "company_id": "other"})
    assert seen == {"query": "vesting", "company_id": "nimbus", "role": "employee", "stakeholder_id": "sh_priya"}


def test_employee_vesting_reads_only_own_grants(grant_calls: list) -> None:
    out = tools_by_name(PRIYA)["get_vesting_status"].invoke({"stakeholder_id": "sh_rahul"})
    assert grant_calls == [("nimbus", "sh_priya")]
    assert '"vested": 2100' in out and '"next_vest_date": "2026-11-01"' in out


def test_vesting_as_of_override(grant_calls: list) -> None:
    out = tools_by_name(PRIYA)["get_vesting_status"].invoke({"as_of": "2026-11-03"})
    assert '"vested": 2200' in out


def test_bad_date_is_an_error_message_not_an_exception(grant_calls: list) -> None:
    assert '"error"' in tools_by_name(PRIYA)["get_vesting_status"].invoke({"as_of": "next month"})


def test_employee_get_grants_reads_only_own(grant_calls: list) -> None:
    out = tools_by_name(PRIYA)["get_grants"].invoke({})
    assert grant_calls == [("nimbus", "sh_priya")]
    assert '"strike_price": 10' in out and "company_id" not in out


def test_admin_resolves_name_within_own_company(grant_calls: list) -> None:
    tools_by_name(ARJUN)["get_grants"].invoke({"stakeholder_name": "rahul"})
    assert grant_calls == [("nimbus", "sh_rahul")]


def test_admin_ambiguous_or_unknown_name_is_an_error(grant_calls: list) -> None:
    tools = tools_by_name(ARJUN)
    assert "several" in tools["get_grants"].invoke({"stakeholder_name": "Ravi"})
    assert "no one" in tools["get_grants"].invoke({"stakeholder_name": "Zed"})
    assert grant_calls == []


# --- formatting for the model ---

def test_search_results_are_tagged_for_citation() -> None:
    chunk = RetrievedChunk("c1", "d", "ESOP Policy", "policy", 5, "6. Exercise Window", "6.1 Ninety days.", 0.6, None)
    text = factory.format_chunks([chunk])
    assert text.startswith("[ESOP Policy, p. 5] 6. Exercise Window\n6.1 Ninety days.")
    assert factory.format_chunks([]) == factory.NO_RESULTS


# --- system prompt ---

def test_prompt_has_date_name_role_and_refusals() -> None:
    text = prompts.build_system_prompt(PRIYA, AS_OF)
    assert "Today is 2026-10-03" in text and "Priya Sharma (employee)" in text
    assert prompts.NOT_FOUND_MESSAGE in text and prompts.ACCESS_DENIED_MESSAGE in text


def test_admin_prompt_has_no_employee_restriction() -> None:
    text = prompts.build_system_prompt(ARJUN, AS_OF)
    assert "Arjun Mehta (admin)" in text and prompts.ACCESS_DENIED_MESSAGE not in text


def test_prompt_defaults_to_today() -> None:
    assert f"Today is {date.today().isoformat()}" in prompts.build_system_prompt(PRIYA)
