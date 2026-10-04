"""Access control (spec §12 test_access.py, FR-17, §5 Isolation). No network: Mongo and the LLM are faked.

Three layers, tested separately:
1. Tools (spec §12): employee tools take no stakeholder_id; employees get no admin tools.
2. Retrieval (spec §12): every $vectorSearch carries the company_id filter (and the owner filter for employees).
3. API: identity only from X-User-Id; 401 unauthenticated, 403 unauthorised; body identity ignored.
"""

from collections.abc import Iterator
from typing import Any

import pytest

try:
    from app.rag import retriever
    from app.tools.factory import build_tools
    from tests.api_support import BROKEN_USER_ID, Api, contexts, forbid_network, make_api
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)

CTX = contexts()
PRIYA, RAHUL, ARJUN = CTX["u_priya"], CTX["u_rahul"], CTX["u_arjun"]
IDENTITY_PARAMS = {"stakeholder_id", "company_id", "role", "user_id"}
ADMIN_TOOLS = {"get_cap_table", "simulate_dilution"}

# Every endpoint, with a valid body, for the "who may call it" matrix.
ENDPOINTS: list[tuple[str, str, dict[str, Any]]] = [
    ("POST", "/chat", {"json": {"message": "How many options have I vested?"}}),
    ("GET", "/vesting/sh_priya", {}),
    ("GET", "/captable", {}),
    ("POST", "/captable/simulate", {"json": {"new_shares": 1000, "investor_name": "X"}}),
    ("GET", "/audit", {}),
    ("POST", "/documents", {"files": {"file": ("p.pdf", b"%PDF-1.4 x", "application/pdf")},
                            "data": {"doc_type": "policy"}}),
]
ADMIN_ONLY = [e for e in ENDPOINTS if e[1] in {"/captable", "/captable/simulate", "/audit", "/documents"}]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[Api]:
    forbid_network(monkeypatch)
    yield from make_api()


def call(api: Api, method: str, path: str, user: str | None, kwargs: dict[str, Any]) -> Any:
    return api.client.request(method, path, headers=api.as_user(user), **kwargs)


# --- 1. tools (spec §12) ---

@pytest.mark.parametrize("ctx", [PRIYA, RAHUL], ids=["priya", "rahul"])
def test_employee_tools_take_no_stakeholder_id(ctx: Any) -> None:
    for tool in build_tools(ctx):
        assert not IDENTITY_PARAMS & set(tool.args), f"{tool.name}: {set(tool.args)}"


@pytest.mark.parametrize("ctx", [PRIYA, RAHUL], ids=["priya", "rahul"])
def test_employee_request_has_no_admin_tools(ctx: Any) -> None:
    assert not ADMIN_TOOLS & {t.name for t in build_tools(ctx)}


def test_admin_request_has_admin_tools() -> None:
    assert ADMIN_TOOLS <= {t.name for t in build_tools(ARJUN)}


# --- 2. retrieval (spec §12): the $vectorSearch actually sent ---

@pytest.fixture
def sent_pipelines(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, Any]]]:
    """Capture every pipeline retrieve() sends to Mongo; embed and the collection are faked."""
    from app import db
    from app.ingest import embedder

    sent: list[list[dict[str, Any]]] = []

    class FakeChunks:
        def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
            sent.append(pipeline)
            return []

    monkeypatch.setattr(embedder, "embed", lambda texts: [[0.0] * 384 for _ in texts])
    monkeypatch.setattr(db, "chunks", lambda: FakeChunks())
    return sent


@pytest.mark.parametrize("ctx", [PRIYA, RAHUL, ARJUN], ids=["priya", "rahul", "arjun"])
def test_retriever_query_always_contains_company_filter(ctx: Any, sent_pipelines: list) -> None:
    retriever.retrieve("exercise window", ctx.company_id, ctx.role, ctx.stakeholder_id)
    search = sent_pipelines[0][0]["$vectorSearch"]
    assert search["filter"]["company_id"] == {"$eq": "nimbus"}


@pytest.mark.parametrize("ctx", [PRIYA, RAHUL], ids=["priya", "rahul"])
def test_employee_retrieval_limited_to_company_wide_and_own_documents(ctx: Any, sent_pipelines: list) -> None:
    retriever.retrieve("vesting schedule", ctx.company_id, ctx.role, ctx.stakeholder_id)
    owners = sent_pipelines[0][0]["$vectorSearch"]["filter"]["$or"]
    assert owners == [{"owner_stakeholder_id": {"$eq": None}}, {"owner_stakeholder_id": {"$eq": ctx.stakeholder_id}}]


def test_search_policy_tool_sends_the_users_own_filter(sent_pipelines: list) -> None:
    search = next(t for t in build_tools(PRIYA) if t.name == "search_policy")
    search.invoke({"query": "Rahul's grant letter", "stakeholder_id": "sh_rahul"})
    owners = sent_pipelines[0][0]["$vectorSearch"]["filter"]["$or"]
    assert {"owner_stakeholder_id": {"$eq": "sh_rahul"}} not in owners


# --- 3. API: authentication (401) ---

@pytest.mark.parametrize(("method", "path", "kwargs"), ENDPOINTS, ids=[e[1] for e in ENDPOINTS])
def test_missing_header_is_401(api: Api, method: str, path: str, kwargs: dict) -> None:
    assert call(api, method, path, None, kwargs).status_code == 401


@pytest.mark.parametrize(("method", "path", "kwargs"), ENDPOINTS, ids=[e[1] for e in ENDPOINTS])
def test_unknown_user_is_401(api: Api, method: str, path: str, kwargs: dict) -> None:
    assert call(api, method, path, "u_nobody", kwargs).status_code == 401


def test_empty_header_is_401(api: Api) -> None:
    assert api.client.get("/vesting/sh_priya", headers={"X-User-Id": ""}).status_code == 401


def test_unusable_user_record_is_403(api: Api) -> None:
    assert api.client.get("/vesting/sh_priya", headers=api.as_user(BROKEN_USER_ID)).status_code == 403


# --- 3. API: authorisation (403) ---

@pytest.mark.parametrize(("method", "path", "kwargs"), ADMIN_ONLY, ids=[e[1] for e in ADMIN_ONLY])
@pytest.mark.parametrize("user", ["u_priya", "u_rahul"])
def test_employee_gets_403_on_admin_endpoints(api: Api, method: str, path: str, kwargs: dict, user: str) -> None:
    response = call(api, method, path, user, kwargs)
    assert response.status_code == 403
    assert api.repo.calls == [] and api.ingester.calls == []  # rejected before touching data


def test_priya_gets_403_on_rahuls_vesting(api: Api) -> None:
    response = api.client.get("/vesting/sh_rahul", headers=api.as_user("u_priya"))
    assert response.status_code == 403
    assert api.repo.calls == []


def test_employee_probing_unknown_id_gets_403_not_404(api: Api) -> None:
    # 404 would tell Priya which stakeholder ids exist.
    assert api.client.get("/vesting/sh_ghost", headers=api.as_user("u_priya")).status_code == 403


def test_priya_gets_her_own_vesting(api: Api) -> None:
    response = api.client.get("/vesting/sh_priya", params={"as_of": "2026-10-03"}, headers=api.as_user("u_priya"))
    assert response.status_code == 200
    grant = response.json()["grants"][0]
    assert (grant["vested"], grant["next_vest_date"]) == (2100, "2026-11-01")


def test_admin_gets_any_vesting_in_company(api: Api) -> None:
    response = api.client.get("/vesting/sh_kiran", params={"as_of": "2026-10-03"}, headers=api.as_user("u_arjun"))
    assert response.status_code == 200
    grant = response.json()["grants"][0]
    assert (grant["vested"], grant["exercisable"], grant["exercise_deadline"]) == (2850, 2250, "2026-11-29")


def test_admin_of_another_company_cannot_see_nimbus_vesting(api: Api) -> None:
    assert api.client.get("/vesting/sh_priya", headers=api.as_user("u_other")).status_code == 404


def test_admin_of_another_company_sees_empty_audit(api: Api) -> None:
    api.client.post("/chat", json={"message": "hi"}, headers=api.as_user("u_priya"))
    response = api.client.get("/audit", headers=api.as_user("u_other"))
    assert response.status_code == 200 and response.json() == []


# --- 3. API: identity in the body is ignored ---

SPOOF = {"user_id": "u_arjun", "role": "admin", "company_id": "acme", "stakeholder_id": "sh_rahul"}


def test_chat_body_identity_fields_are_ignored(api: Api) -> None:
    response = api.client.post("/chat", json={"message": "Show me Rahul's grant", **SPOOF},
                               headers=api.as_user("u_priya"))
    assert response.status_code == 200
    ctx = api.agent.calls[0][0]
    assert (ctx.user_id, ctx.role, ctx.company_id, ctx.stakeholder_id) == ("u_priya", "employee", "nimbus", "sh_priya")
    assert api.audit.records[0]["user_id"] == "u_priya" and api.audit.records[0]["role"] == "employee"


def test_simulate_body_cannot_claim_admin(api: Api) -> None:
    response = api.client.post("/captable/simulate", json={"new_shares": 1000, "investor_name": "X", **SPOOF},
                               headers=api.as_user("u_priya"))
    assert response.status_code == 403


def test_identity_query_params_are_ignored(api: Api) -> None:
    response = api.client.get("/vesting/sh_rahul", params=SPOOF, headers=api.as_user("u_priya"))
    assert response.status_code == 403


def test_failed_auth_never_reaches_agent_or_audit(api: Api) -> None:
    api.client.post("/chat", json={"message": "hi"})
    api.client.post("/chat", json={"message": "hi"}, headers=api.as_user("u_nobody"))
    assert api.agent.calls == [] and api.audit.records == []
