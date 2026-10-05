"""Agent loop tests with a scripted fake model: cap, not-found short-circuit, citations. No LLM, no Mongo."""

from datetime import date
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

try:
    from app import agent
    from app.context import RequestContext
    from app.rag.prompts import NOT_FOUND_MESSAGE
    from app.rag.retriever import RetrievedChunk
    from app.tools import factory
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)

PRIYA = RequestContext("u_priya", "employee", "nimbus", "sh_priya", "Priya Sharma")
AS_OF = date(2026, 10, 3)
P5 = RetrievedChunk("pol_012", "pol", "ESOP Policy", "policy", 5, "6. Exercise Window After Leaving",
                    "6.1 Exercise Window. ninety (90) days ...", 0.57, None)
P5_WEAKER = RetrievedChunk("pol_013", "pol", "ESOP Policy", "policy", 5, "6. Exercise Window After Leaving",
                           "6.5 Bad Leavers ...", 0.41, None)


class ScriptedModel:
    """Stands in for a chat model: returns the scripted replies in order and records each request."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = list(replies)
        self.requests: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.requests.append(list(messages))
        return self.replies.pop(0)


def call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def use_model(monkeypatch: pytest.MonkeyPatch):
    """Install a ScriptedModel and fake retrieval returning `chunks` (default: none)."""

    def install(replies: list[AIMessage], chunks: list[RetrievedChunk] | None = None) -> ScriptedModel:
        model = ScriptedModel(replies)
        monkeypatch.setattr(agent, "get_chat_model", lambda: model)
        monkeypatch.setattr(factory, "retrieve", lambda *a, **k: list(chunks or []))
        monkeypatch.setattr(factory.repo, "get_grants", lambda *a, **k: [])
        return model

    return install


def test_plain_answer_without_tools(use_model) -> None:
    model = use_model([AIMessage(content="I can only share information about your own grants.")])
    out = agent.run_agent(PRIYA, "Show me Rahul's grant.", as_of=AS_OF)
    assert out["tool_calls"] == [] and out["citations"] == []
    assert len(model.requests) == 1


def test_tool_result_goes_back_to_model_and_citation_is_built(use_model) -> None:
    model = use_model([
        AIMessage(content="", tool_calls=[call("search_policy", {"query": "exercise window after leaving"}, "1")]),
        AIMessage(content="You have 90 days [ESOP Policy, p. 5]."),
    ], chunks=[P5_WEAKER, P5])
    out = agent.run_agent(PRIYA, "How long to exercise after I leave?", as_of=AS_OF)
    second_request = model.requests[1]
    assert isinstance(second_request[-1], ToolMessage) and "[ESOP Policy, p. 5]" in second_request[-1].content
    assert out["tool_calls"] == [{"name": "search_policy", "args": {"query": "exercise window after leaving"}}]
    assert [c["chunk_id"] for c in out["citations"]] == ["pol_012"]  # best-scoring chunk on the page
    assert out["chunk_ids"] == ["pol_013", "pol_012"]


def test_cap_stops_after_four_tool_calls(use_model) -> None:
    greedy = [AIMessage(content="", tool_calls=[call("get_grants", {}, f"{r}-{i}") for i in range(3)])
              for r in range(3)]
    model = use_model([*greedy, AIMessage(content="Done.")])
    out = agent.run_agent(PRIYA, "loop please", as_of=AS_OF)
    assert len(out["tool_calls"]) == agent.MAX_TOOL_CALLS
    # Calls past the cap still get a ToolMessage (providers require one per call), saying it wasn't run.
    limited = [m for m in model.requests[-1] if isinstance(m, ToolMessage) and m.content == agent.LIMIT_REACHED]
    assert len(limited) == 3 * 3 - agent.MAX_TOOL_CALLS
    assert out["answer"] == "Done."


def test_model_that_never_stops_gets_fallback_answer(use_model) -> None:
    replies = [AIMessage(content="", tool_calls=[call("get_grants", {}, str(i))]) for i in range(agent.MAX_MODEL_CALLS)]
    use_model(replies)
    assert agent.run_agent(PRIYA, "loop", as_of=AS_OF)["answer"] == agent.NO_ANSWER


def test_empty_policy_search_returns_not_found_without_second_llm_call(use_model) -> None:
    model = use_model([AIMessage(content="", tool_calls=[call("search_policy", {"query": "capital of France"}, "1")])])
    out = agent.run_agent(PRIYA, "What is the capital of France?", as_of=AS_OF)
    assert out["answer"] == NOT_FOUND_MESSAGE
    assert len(model.requests) == 1


def test_empty_search_plus_data_tool_is_not_short_circuited(use_model) -> None:
    model = use_model([
        AIMessage(content="", tool_calls=[call("search_policy", {"query": "x"}, "1"), call("get_grants", {}, "2")]),
        AIMessage(content="Your grant is 4,800 options."),
    ])
    out = agent.run_agent(PRIYA, "my grant?", as_of=AS_OF)
    assert out["answer"] == "Your grant is 4,800 options."
    assert len(model.requests) == 2


def test_unknown_tool_is_reported_to_model_not_run(use_model) -> None:
    model = use_model([
        AIMessage(content="", tool_calls=[call("get_cap_table", {}, "1")]),
        AIMessage(content="I can only share information about your own grants."),
    ])
    agent.run_agent(PRIYA, "show the cap table", as_of=AS_OF)
    reply = model.requests[1][-1]
    assert isinstance(reply, ToolMessage) and "no tool named 'get_cap_table'" in reply.content


# --- citation parsing (pure) ---

@pytest.mark.parametrize(("answer", "tags"), [
    ("90 days [ESOP Policy, p. 5].", [("ESOP Policy", 5)]),
    ("[ESOP Policy, p. 6; Grant Letter: Priya Sharma, p. 2]", [("ESOP Policy", 6), ("Grant Letter: Priya Sharma", 2)]),
    ("2,200 [ESOP Policy, p.6; get_vesting_status]", [("ESOP Policy", 6)]),
    ("no tags here", []),
])
def test_cited_tags(answer: str, tags: list[tuple[str, int]]) -> None:
    assert agent.cited_tags(answer) == tags


def test_lenticular_brackets_are_citations() -> None:
    answer = "lapse 【ESOP Policy, p. 6】 and 90 days 【ESOP Policy, p. 5; Grant Letter: Priya Sharma, p. 2】"
    assert agent.cited_tags(answer) == [("ESOP Policy", 6), ("ESOP Policy", 5), ("Grant Letter: Priya Sharma", 2)]


# --- answer normalisation (gpt-oss on Groq emits these characters) ---

@pytest.mark.parametrize(("raw", "clean"), [
    ("50 %", "50 %"),                           # narrow no-break space
    ("90 days", "90 days"),                     # no-break space
    ("2026‑11‑03", "2026-11-03"),          # non-breaking hyphen
    ("1 000 options", "1 000 options"),         # thin space
    ("2026‐11‐03", "2026-11-03"),          # plain HYPHEN, what NFKC makes of U+2011
    ("５０％", "50%"),                   # full-width digits and % (NFKC)
    ("[ESOP Policy, p. 7]", "[ESOP Policy, p. 7]"),
    ("as of that date 【​】.", "as of that date."),  # empty marker seen in run 5
    ("keep [ ] these", "keep these"),
    ("zero​width", "zerowidth"),
    ("plain ascii, unchanged", "plain ascii, unchanged"),
])
def test_normalize_answer(raw: str, clean: str) -> None:
    assert agent.normalize_answer(raw) == clean


def test_returned_answer_is_normalised_and_its_citation_parsed(use_model) -> None:
    use_model([
        AIMessage(content="", tool_calls=[call("search_policy", {"query": "exercise window"}, "1")]),
        AIMessage(content="Within 90 days 【ESOP Policy, p. 5】."),
    ], chunks=[P5])
    out = agent.run_agent(PRIYA, "exercise window?", as_of=AS_OF)
    assert out["answer"] == "Within 90 days [ESOP Policy, p. 5]."  # 【...】 rewritten to [...]
    assert [(c["doc_title"], c["page"]) for c in out["citations"]] == [("ESOP Policy", 5)]


def test_invented_citation_is_dropped() -> None:
    cites = agent.build_citations("[ESOP Policy, p. 5] and [ESOP Policy, p. 9]", [P5])
    assert [(c["doc_title"], c["page"]) for c in cites] == [("ESOP Policy", 5)]


def test_history_keeps_last_six() -> None:
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(10)]
    messages = agent.to_messages(history)
    assert [m.content for m in messages] == [str(i) for i in range(4, 10)]


# --- citation validation (grounding bug: Rahul's answer cited p. 7, only p. 5 and p. 6 were retrieved) ---

P6 = RetrievedChunk("pol_014", "pol", "ESOP Policy", "policy", 6, "7. Termination of Employment",
                    "7.1 Unvested Options lapse on the Last Working Day ...", 0.52, None)
SEARCH = AIMessage(content="", tool_calls=[call("search_policy", {"query": "exercise window after leaving"}, "s1")])
BAD = "You keep vested options [ESOP Policy, p. 6] and have 90 days [ESOP Policy, p. 7]."
GOOD = "You keep vested options [ESOP Policy, p. 6] and have 90 days [ESOP Policy, p. 5]."


def test_valid_citations_need_no_retry(use_model) -> None:
    model = use_model([SEARCH, AIMessage(content=GOOD)], chunks=[P5, P6])
    out = agent.run_agent(PRIYA, "leave?", as_of=AS_OF)
    assert out["answer"] == GOOD and out["flags"] == []
    assert out["citation_check"] == {"total": 2, "invalid": 0, "retried": False, "stripped": 0}
    assert len(model.requests) == 2  # no corrective call


def test_invalid_citation_is_fixed_by_one_corrective_retry(use_model) -> None:
    model = use_model([SEARCH, AIMessage(content=BAD), AIMessage(content=GOOD)], chunks=[P5, P6])
    out = agent.run_agent(PRIYA, "leave?", as_of=AS_OF)
    assert out["answer"] == GOOD and out["flags"] == [agent.FLAG_CITATION_RETRIED]
    assert out["citation_check"] == {"total": 2, "invalid": 1, "retried": True, "stripped": 0}  # first draft
    correction = model.requests[2][-1]
    assert isinstance(correction, HumanMessage)
    assert "[ESOP Policy, p. 7]" in correction.content
    assert "[ESOP Policy, p. 5], [ESOP Policy, p. 6]" in correction.content  # the valid sources, listed
    assert [(c["doc_title"], c["page"]) for c in out["citations"]] == [("ESOP Policy", 6), ("ESOP Policy", 5)]


def test_still_invalid_after_retry_is_stripped_and_flagged(use_model) -> None:
    model = use_model([SEARCH, AIMessage(content=BAD), AIMessage(content=BAD)], chunks=[P5, P6])
    out = agent.run_agent(PRIYA, "leave?", as_of=AS_OF)
    assert out["answer"] == "You keep vested options [ESOP Policy, p. 6] and have 90 days."
    assert out["flags"] == [agent.FLAG_CITATION_RETRIED, agent.FLAG_CITATION_INVALID]
    assert out["citation_check"] == {"total": 2, "invalid": 1, "retried": True, "stripped": 1}
    assert len(model.requests) == 3  # exactly one retry


def test_retry_that_asks_for_tools_keeps_the_draft_and_strips(use_model) -> None:
    use_model([SEARCH, AIMessage(content=BAD), AIMessage(content="", tool_calls=[call("get_grants", {}, "g")])],
              chunks=[P5, P6])
    out = agent.run_agent(PRIYA, "leave?", as_of=AS_OF)
    assert out["answer"] == "You keep vested options [ESOP Policy, p. 6] and have 90 days."
    assert agent.FLAG_CITATION_INVALID in out["flags"]


def test_citation_without_any_search_is_invalid(use_model) -> None:
    model = use_model([AIMessage(content="Options vest monthly [ESOP Policy, p. 3]."),
                       AIMessage(content="Options vest monthly [ESOP Policy, p. 3].")])
    out = agent.run_agent(PRIYA, "how do options vest?", as_of=AS_OF)
    assert out["answer"] == "Options vest monthly."
    assert "must not contain any [Title, p. N] citation" in model.requests[1][-1].content


def test_lenticular_invalid_citation_is_caught_after_rewrite(use_model) -> None:
    use_model([SEARCH, AIMessage(content="90 days 【ESOP Policy, p. 7】."), AIMessage(content="90 days 【ESOP Policy, p. 5】.")],
              chunks=[P5])
    out = agent.run_agent(PRIYA, "window?", as_of=AS_OF)
    assert out["answer"] == "90 days [ESOP Policy, p. 5]." and out["flags"] == [agent.FLAG_CITATION_RETRIED]


@pytest.mark.parametrize(("text", "invalid", "expected"), [
    ("a [ESOP Policy, p. 7].", {("ESOP Policy", 7)}, "a."),
    ("a [ESOP Policy, p. 6; ESOP Policy, p. 7] b", {("ESOP Policy", 7)}, "a [ESOP Policy, p. 6] b"),
    ("a [ESOP Policy, p. 6] b", {("ESOP Policy", 7)}, "a [ESOP Policy, p. 6] b"),
    ("see [get_vesting_status] and [ESOP Policy, p.7]", {("ESOP Policy", 7)}, "see [get_vesting_status] and"),
])
def test_strip_citations(text: str, invalid: set, expected: str) -> None:
    assert agent.strip_citations(text, invalid) == expected


def test_normalize_rewrites_lenticular_brackets() -> None:
    assert agent.normalize_answer("90 days 【ESOP Policy, p. 5】 and 【Grant Letter: Priya Sharma, p. 2】") == \
        "90 days [ESOP Policy, p. 5] and [Grant Letter: Priya Sharma, p. 2]"
