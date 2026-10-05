"""UI tests (FR-19, FR-9): the API client's error handling, and the Streamlit app run headlessly with AppTest.

No server, no Mongo, no LLM: `requests.request` is replaced by a fake that
records every call, so these tests also prove the UI only ever talks HTTP to API_URL.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests

try:
    from streamlit.testing.v1 import AppTest

    from app.config import settings
    from app.rag.prompts import ACCESS_DENIED_MESSAGE, NOT_FOUND_MESSAGE
    from ui import api_client
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")
BASE = "http://api.test"

ANSWER = {
    "answer": "You keep 2,200 vested options and have 90 days to exercise them [ESOP Policy, p. 5].",
    "citations": [{"doc_title": "ESOP Policy", "page": 5, "section": "6. Exercise Window After Leaving",
                   "snippet": "6.1 Exercise Window. An Employee who leaves", "chunk_id": "c12"}],
    "tool_calls": [{"name": "get_vesting_status", "args": {"as_of": "2026-11-03"}},
                   {"name": "search_policy", "args": {"query": "exercise window after leaving, termination"}}],
    "latency_ms": 2674,
    "outcome": "answered",
}
CAP_TABLE = {
    "rows": [{"stakeholder_id": "sh_arjun", "name": "Arjun Mehta", "shares": 6_000_000, "outstanding_options": 0,
              "fully_diluted": 6_000_000, "issued_pct": 63.15, "fully_diluted_pct": 57.14},
             {"stakeholder_id": None, "name": "Unallocated ESOP pool", "shares": 0, "outstanding_options": 0,
              "fully_diluted": 989_950, "issued_pct": 0.0, "fully_diluted_pct": 9.43}],
    "total_issued": 9_500_600, "total_fully_diluted": 10_500_000,
    "pool": {"pool_size": 1_000_000, "outstanding": 9_450, "exercised": 600, "lapsed": 750, "unallocated": 989_950},
}
DILUTION = {
    "investor_name": "Horizon Capital", "new_shares": 2_000_000,
    "rows": [{"stakeholder_id": "sh_arjun", "name": "Arjun Mehta", "shares": 6_000_000, "outstanding_options": 0,
              "issued_pct_before": 63.15, "issued_pct_after": 52.17,
              "fully_diluted_pct_before": 57.14, "fully_diluted_pct_after": 48.0}],
    "total_issued_before": 9_500_600, "total_issued_after": 11_500_600,
    "total_fully_diluted_before": 10_500_000, "total_fully_diluted_after": 12_500_000,
}


class FakeResponse:
    """Just enough of requests.Response."""

    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None, text: bool = False):
        self.status_code, self._body, self.headers, self._text = status, body, headers or {}, text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        if self._text:
            raise ValueError("not JSON")
        return self._body


class FakeHttp:
    """Replaces requests.request: records calls; responds by (method, path) or raises."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], FakeResponse | Exception] = {
            ("POST", "/chat"): FakeResponse(200, ANSWER, {"X-Audit-Id": "audit_1"}),
            ("GET", "/captable"): FakeResponse(200, CAP_TABLE),
            ("POST", "/captable/simulate"): FakeResponse(200, DILUTION),
            ("GET", "/audit"): FakeResponse(200, []),
        }

    def __call__(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        path = url.split("://", 1)[1].split("/", 1)[1]
        self.calls.append({"method": method, "url": url, "path": "/" + path, **kwargs})
        outcome = self.routes[(method, "/" + path)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def paths(self) -> list[str]:
        return [c["path"] for c in self.calls]


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
    fake = FakeHttp()
    monkeypatch.setattr(api_client.requests, "request", fake)
    return fake


# --- api_client ---

def test_call_sends_user_header_and_timeout(http: FakeHttp) -> None:
    result = api_client.chat(BASE, "u_priya", "hi", [], date(2026, 10, 3))
    call = http.calls[0]
    assert call["headers"] == {"X-User-Id": "u_priya"} and call["timeout"] == 60
    assert call["json"] == {"message": "hi", "history": [], "as_of": "2026-10-03"}
    assert result.ok and result.audit_id == "audit_1" and result.data["latency_ms"] == 2674


@pytest.mark.parametrize(("raised", "fragment"), [
    (requests.Timeout(), "No answer within 60 s"),
    (requests.ConnectionError(), "Can't reach the API at http://api.test"),
    (requests.TooManyRedirects(), "Request failed: TooManyRedirects"),
])
def test_network_failures_become_messages(http: FakeHttp, raised: Exception, fragment: str) -> None:
    http.routes[("GET", "/captable")] = raised
    result = api_client.cap_table(BASE, "u_arjun")
    assert not result.ok and fragment in result.error


@pytest.mark.parametrize(("status", "body", "fragment"), [
    (401, {"detail": "Unknown user"}, "not signed in"),
    (403, {"detail": "Admin only"}, "don't have access to this. Admin only"),
    (404, {"detail": "Stakeholder not found"}, "Not found"),
    (422, {"detail": [{"loc": ["body", "new_shares"], "msg": "must be greater than 0"}]},
     "new_shares: must be greater than 0"),
    (500, {"detail": "Something went wrong while answering. The error has been logged."}, "has been logged"),
    (502, None, "server had a problem"),
])
def test_http_errors_become_messages(http: FakeHttp, status: int, body: Any, fragment: str) -> None:
    http.routes[("GET", "/captable")] = FakeResponse(status, body)
    result = api_client.cap_table(BASE, "u_priya")
    assert not result.ok and result.status == status and fragment in result.error


def test_non_json_success_is_an_error(http: FakeHttp) -> None:
    http.routes[("GET", "/audit")] = FakeResponse(200, text=True)
    assert "isn't JSON" in api_client.audit(BASE, "u_arjun").error


def test_audit_asks_for_20(http: FakeHttp) -> None:
    api_client.audit(BASE, "u_arjun")
    assert http.calls[0]["params"] == {"limit": 20}


# --- the Streamlit app, headless ---

def run_app(http: FakeHttp, user: str = "u_priya") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["user_id"] = user
    return at.run()


def ask(at: AppTest, text: str) -> AppTest:
    at.chat_input[0].set_value(text)
    return at.run()


def test_employee_sees_chat_only(http: FakeHttp) -> None:
    at = run_app(http)
    assert not at.exception
    assert len(at.tabs) == 0 and len(at.chat_input) == 1
    assert http.calls == []  # nothing fetched until the user asks


def test_admin_sees_admin_tabs(http: FakeHttp) -> None:
    at = run_app(http, "u_arjun")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Chat", "Cap table", "Dilution", "Audit log"]
    assert http.calls == []  # lazy tabs: only the open Chat tab ran


def test_mixed_question_shows_answer_citations_and_caption(http: FakeHttp) -> None:
    at = ask(run_app(http), "If I leave next month, how many options do I keep?")
    assert not at.exception
    call = http.calls[0]
    assert call["url"] == f"{settings.api_url}/chat" and call["headers"] == {"X-User-Id": "u_priya"}
    assert call["json"]["as_of"] == "2026-10-03"
    assert any("2,200" in m.value for m in at.markdown)
    assert [e.label for e in at.expander] == ["ESOP Policy, p. 5 · 6. Exercise Window After Leaving"]
    [caption] = [c.value for c in at.main.caption if c.value.startswith("Tools:")]
    assert "get_vesting_status(as_of=2026-11-03)" in caption and "2.7 s" in caption and "audit audit_1" in caption


def test_refusal_is_calm_info_not_error(http: FakeHttp) -> None:
    body = {"answer": ACCESS_DENIED_MESSAGE, "outcome": "refused", "citations": [], "tool_calls": [], "latency_ms": 750}
    http.routes[("POST", "/chat")] = FakeResponse(200, body, {"X-Audit-Id": "a2"})
    at = ask(run_app(http), "Show me Rahul's grant.")
    assert [i.value for i in at.info] == [ACCESS_DENIED_MESSAGE] and len(at.error) == 0


def test_not_found_is_calm_info(http: FakeHttp) -> None:
    body = {"answer": NOT_FOUND_MESSAGE, "outcome": "not_found", "citations": [], "tool_calls": [], "latency_ms": 900}
    http.routes[("POST", "/chat")] = FakeResponse(200, body, {"X-Audit-Id": "a3"})
    at = ask(run_app(http), "What is Nimbus's valuation?")
    assert [i.value for i in at.info] == [NOT_FOUND_MESSAGE]


def test_styling_follows_outcome_not_answer_text(http: FakeHttp) -> None:
    # A reworded refusal is still styled as a refusal; refusal *text* with outcome "answered" is not.
    reworded = {"answer": "Sorry, that's another employee's grant.", "outcome": "refused", "citations": [],
                "tool_calls": [], "latency_ms": 500}
    http.routes[("POST", "/chat")] = FakeResponse(200, reworded, {"X-Audit-Id": "a4"})
    at = ask(run_app(http), "Show me Rahul's grant.")
    assert [i.value for i in at.info] == ["Sorry, that's another employee's grant."]
    plain = {"answer": NOT_FOUND_MESSAGE, "outcome": "answered", "citations": [], "tool_calls": [], "latency_ms": 500}
    http.routes[("POST", "/chat")] = FakeResponse(200, plain, {"X-Audit-Id": "a5"})
    at = ask(at, "anything")
    assert len(at.info) == 1  # only the earlier refusal


def test_examples_shown_per_role_and_clicking_sends(http: FakeHttp) -> None:
    at = run_app(http)
    employee = [b.label for b in at.button if b.key and b.key.startswith("example_")]
    assert 3 <= len(employee) <= 4 and any("leave" in q for q in employee) and any("acquired" in q for q in employee)
    next(b for b in at.button if b.label == employee[2]).click().run()
    assert http.calls[-1]["json"]["message"] == employee[2]
    assert http.calls[-1]["headers"] == {"X-User-Id": "u_priya"}
    assert not [b for b in at.button if b.key and b.key.startswith("example_")]  # gone once the chat starts
    admin = [b.label for b in run_app(FakeHttp(), "u_arjun").button if b.key and b.key.startswith("example_")]
    assert any("2,000,000" in q for q in admin) and any("Kiran" in q for q in admin)


def test_switching_user_drops_a_queued_example(http: FakeHttp) -> None:
    at = run_app(http)
    at.session_state["pending_prompt"] = "How many of my options have vested so far?"
    at.selectbox[0].set_value("u_rahul").run()
    assert http.calls == []


def test_api_error_shows_message_not_traceback(http: FakeHttp) -> None:
    http.routes[("POST", "/chat")] = requests.Timeout()
    at = ask(run_app(http), "How many options have I vested?")
    assert not at.exception
    assert "No answer within 60 s" in at.error[0].value


def test_history_is_sent_and_errors_are_left_out(http: FakeHttp) -> None:
    at = ask(run_app(http), "first question")
    http.routes[("POST", "/chat")] = requests.ConnectionError()
    at = ask(at, "second question")
    http.routes[("POST", "/chat")] = FakeResponse(200, ANSWER, {"X-Audit-Id": "a9"})
    ask(at, "third question")
    history = http.calls[-1]["json"]["history"]
    assert [h["content"] for h in history] == ["first question", ANSWER["answer"], "second question"]


def test_switching_user_clears_conversation(http: FakeHttp) -> None:
    at = ask(run_app(http), "How many options have I vested?")
    assert len(at.session_state["messages"]) == 2
    at.selectbox[0].set_value("u_rahul").run()
    assert at.session_state["messages"] == [] and len(at.chat_message) == 0


def test_as_of_picker_is_sent(http: FakeHttp) -> None:
    at = run_app(http)
    at.date_input[0].set_value(date(2027, 1, 1)).run()
    ask(at, "How many options have I vested?")
    assert http.calls[-1]["json"]["as_of"] == "2027-01-01"


def open_tab(at: AppTest, label: str) -> AppTest:
    """Select an admin tab (lazy tabs rerun on change).

    AppTest (Streamlit 1.65) has no way to click a tab, and on each run it sends the tab
    widget's last value back, so the selection is set through session state before *every*
    run, as a browser would send it.
    """
    at.session_state["admin_tab"] = label
    return at.run()


def test_cap_table_tab(http: FakeHttp) -> None:
    at = open_tab(run_app(http, "u_arjun"), "Cap table")
    assert not at.exception and http.paths() == ["/captable"]
    assert [m.value for m in at.metric] == ["1,000,000", "9,450", "600", "750", "989,950"]
    assert at.dataframe[0].value["fully_diluted_pct"].tolist() == [57.14, 9.43]


def test_dilution_tab(http: FakeHttp) -> None:
    at = open_tab(run_app(http, "u_arjun"), "Dilution")
    # The form's Simulate button (defaults: 2,000,000 to Horizon Capital); the sidebar has buttons too.
    next(b for b in at.button if b.label == "Simulate").click()
    at = open_tab(at, "Dilution")  # keep the tab selected on the click's rerun (see open_tab)
    assert not at.exception  # regression: the result once shared the form's session-state key and crashed
    call = http.calls[-1]
    assert call["path"] == "/captable/simulate"
    assert call["json"] == {"new_shares": 2_000_000, "investor_name": "Horizon Capital"}
    assert at.dataframe[0].value["fully_diluted_pct_after"].tolist() == [48.0]


def test_audit_tab_asks_for_latest_20(http: FakeHttp) -> None:
    at = open_tab(run_app(http, "u_arjun"), "Audit log")
    assert not at.exception and http.calls[-1]["params"] == {"limit": 20}


def test_admin_tab_error_is_a_message(http: FakeHttp) -> None:
    http.routes[("GET", "/captable")] = FakeResponse(403, {"detail": "Admin only"})
    at = open_tab(run_app(http, "u_arjun"), "Cap table")
    assert not at.exception and "don't have access" in at.error[0].value


def test_ui_never_imports_database_or_llm_layers() -> None:
    source = Path(APP).read_text(encoding="utf-8") + (Path(APP).parent / "api_client.py").read_text(encoding="utf-8")
    for forbidden in ("app.db", "app.agent", "app.llm", "app.tools", "app.rag", "pymongo", "langchain"):
        assert forbidden not in source, forbidden
