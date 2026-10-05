"""Tool-calling agent: one chat turn (spec §6 request flow, §8.6, FR-14, FR-15).

The loop, written out so every step is visible:
  1. send system prompt + history + question to the model, with the user's tools bound;
  2. if the reply contains tool calls, run them (our code, with ctx captured in the tools)
     and append each result as a ToolMessage;
  3. repeat until the model answers in plain text, or the cap of MAX_TOOL_CALLS is hit.

Citations are built from the chunks search_policy actually returned, keeping only the
ones whose [title, p. N] tag appears in the answer. A tag the model made up matches no
retrieved chunk, so it never becomes a citation.

Citation validation (grounding): the in-text tags themselves are also checked against the
(doc_title, page) pairs retrieved in this turn. An invalid tag triggers one corrective
retry listing the valid sources; if the retry is still invalid, the invalid tags are
stripped from the answer and the turn is flagged "citation_invalid" for the audit log.
"""

import re
import unicodedata
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.context import RequestContext
from app.llm import get_chat_model
from app.rag.prompts import NOT_FOUND_MESSAGE, build_system_prompt
from app.rag.retriever import RetrievedChunk
from app.tools.factory import SEARCH_POLICY, build_tools

MAX_TOOL_CALLS = 4  # spec §8.6: per turn, to stop loops
MAX_MODEL_CALLS = MAX_TOOL_CALLS + 2  # each round runs >= 1 tool, plus the answer and one spare
HISTORY_MESSAGES = 6  # FR-16
SNIPPET_CHARS = 300
# [ESOP Policy, p. 6] or the 【ESOP Policy, p. 6】 brackets gpt-oss sometimes uses.
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]|【([^【】]+)】")
# Applied after NFKC. NFKC already turns the three spaces into " ", but it maps the
# non-breaking hyphen U+2011 to U+2010 (HYPHEN), not "-", so both hyphens are listed.
SPACE_AND_HYPHEN = str.maketrans({
    " ": " ",  # no-break space
    " ": " ",  # narrow no-break space ("50 %", "90 days" from gpt-oss)
    " ": " ",  # thin space
    "‑": "-",  # non-breaking hyphen ("2026‑11‑03")
    "‐": "-",  # hyphen: what NFKC turns U+2011 into
    "​": "",   # zero-width space (gpt-oss wrote an empty "【<U+200B>】")
})
EMPTY_BRACKETS_RE = re.compile(r" ?(\[\s*\]|【\s*】)")  # citation markers with nothing inside
LENTICULAR_RE = re.compile(r"【([^【】]*)】")  # gpt-oss's 【...】 -> [...]
SQUARE_RE = re.compile(r"(\s?)\[([^\[\]]+)\]")  # a [...] group and the space before it
TAG_RE = re.compile(r"^(.+?),\s*p\.\s*(\d+)$")  # "ESOP Policy, p. 6"

FLAG_CITATION_RETRIED = "citation_retried"  # the first draft cited an unretrieved page; a corrective retry ran
FLAG_CITATION_INVALID = "citation_invalid"  # still invalid after the retry; the invalid tags were stripped

LIMIT_REACHED = (f"Not run: the limit of {MAX_TOOL_CALLS} tool calls per question was reached. "
                 "Answer now with the information you already have.")
NO_ANSWER = "Sorry, I couldn't complete this answer. Please try rephrasing the question."


def to_messages(history: list[dict[str, str]] | None) -> list[BaseMessage]:
    """Last HISTORY_MESSAGES turns as LangChain messages; history items are {"role", "content"}."""
    out: list[BaseMessage] = []
    for item in (history or [])[-HISTORY_MESSAGES:]:
        cls = HumanMessage if item["role"] == "user" else AIMessage
        out.append(cls(content=item["content"]))
    return out


def run_tool(tools: dict[str, BaseTool], call: dict[str, Any]) -> ToolMessage:
    """Run one tool call from the model; errors go back to the model as text, never crash the turn."""
    tool = tools.get(call["name"])
    if tool is None:  # e.g. an employee's model asking for get_cap_table: it simply doesn't exist
        return ToolMessage(f"Error: no tool named {call['name']!r} is available to this user.",
                           tool_call_id=call["id"], name=call["name"])
    try:
        return tool.invoke(call)  # a ToolCall in -> a ToolMessage out (with .artifact for search_policy)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model as data
        return ToolMessage(f"Error running {call['name']}: {exc}", tool_call_id=call["id"], name=call["name"])


def normalize_answer(text: str) -> str:
    """Unicode NFKC, then plain spaces and "-" for the no-break spaces and hyphens models emit;
    zero-width spaces and empty citation brackets are removed, and 【...】 becomes [...].

    Example: "50<U+202F>% within 90<U+00A0>days of 2026<U+2011>11<U+2011>03 【ESOP Policy, p. 5】"
          -> "50 % within 90 days of 2026-11-03 [ESOP Policy, p. 5]"
    """
    text = unicodedata.normalize("NFKC", text).translate(SPACE_AND_HYPHEN)
    text = EMPTY_BRACKETS_RE.sub("", text)
    return LENTICULAR_RE.sub(r"[\1]", text)


def cited_tags(answer: str) -> list[tuple[str, int]]:
    """(title, page) for every tag in the answer, in order.

    Handles single tags "[ESOP Policy, p. 6]" (or "【ESOP Policy, p. 6】") and combined
    ones, "[ESOP Policy, p. 6; Grant Letter: Priya Sharma, p. 2]". Parts that aren't
    "title, p. N" (e.g. "[get_vesting_status]") are ignored.
    """
    tags = []
    for square, lenticular in BRACKET_RE.findall(answer):
        for part in (square or lenticular).split(";"):
            match = TAG_RE.match(part.strip())
            if match:
                tags.append((match.group(1).strip(), int(match.group(2))))
    return tags


def build_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """One citation per [title, p. N] cited in the answer that matches a retrieved chunk (best score wins)."""
    best: dict[tuple[str, int], RetrievedChunk] = {}
    for c in chunks:
        key = (c.doc_title, c.page)
        if key not in best or c.score > best[key].score:
            best[key] = c
    citations, seen = [], set()
    for key in cited_tags(answer):
        if key in best and key not in seen:
            seen.add(key)
            c = best[key]
            citations.append({"doc_title": c.doc_title, "page": c.page, "section": c.section,
                              "snippet": c.text[:SNIPPET_CHARS], "chunk_id": c.chunk_id})
    return citations


def tag(key: tuple[str, int]) -> str:
    """("ESOP Policy", 6) -> "[ESOP Policy, p. 6]"."""
    return f"[{key[0]}, p. {key[1]}]"


def invalid_tags(answer: str, valid: set[tuple[str, int]]) -> list[tuple[str, int]]:
    """Cited (title, page) pairs that were not retrieved this turn, unique, in order of appearance."""
    return list(dict.fromkeys(key for key in cited_tags(answer) if key not in valid))


def strip_citations(answer: str, invalid: set[tuple[str, int]]) -> str:
    """Remove invalid tags from the (normalised, square-bracket) answer; keep valid tags and other brackets.

    "[ESOP Policy, p. 6; ESOP Policy, p. 7]" with p. 7 invalid -> "[ESOP Policy, p. 6]";
    " [ESOP Policy, p. 7]" alone -> "" (with its leading space).
    """
    def keep(part: str) -> bool:
        match = TAG_RE.match(part.strip())
        return not match or (match.group(1).strip(), int(match.group(2))) not in invalid

    def rewrite(match: re.Match[str]) -> str:
        space, inner = match.group(1), match.group(2)
        parts = [p.strip() for p in inner.split(";")]
        kept = [p for p in parts if keep(p)]
        if len(kept) == len(parts):
            return match.group(0)
        return f"{space}[{'; '.join(kept)}]" if kept else ""

    return SQUARE_RE.sub(rewrite, answer)


def correction_message(invalid: list[tuple[str, int]], valid: set[tuple[str, int]]) -> str:
    """The corrective user turn for the single retry: what was wrong and the only sources allowed."""
    bad = ", ".join(tag(k) for k in invalid)
    if valid:
        allowed = "The only valid sources are: " + ", ".join(tag(k) for k in sorted(valid)) + "."
    else:
        allowed = "No documents were retrieved, so the answer must not contain any [Title, p. N] citation."
    return (f"Citation check: your answer cites {bad}, which search_policy did not return in this conversation. "
            f"{allowed} Rewrite your answer: cite each fact only with the source it actually came from, and "
            "remove any statement you cannot support with those sources. Keep every number from the tool "
            "results unchanged. Do not call tools. Reply with the corrected answer only.")


def validate_citations(
    model: Any, messages: list[BaseMessage], answer: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[str], dict[str, Any]]:
    """Check in-text citations against what was retrieved; one corrective retry; strip what's still invalid.

    Returns (answer, flags, check). `check` describes the *first draft* (total and invalid in-text
    citations, for Phase 8's citation-precision metric), whether a retry ran, and how many tags
    were stripped from the final answer.
    """
    valid = {(c.doc_title, c.page) for c in chunks}
    answer = normalize_answer(answer)
    invalid = invalid_tags(answer, valid)
    check = {"total": len(cited_tags(answer)), "invalid": len(invalid), "retried": False, "stripped": 0}
    if not invalid:
        return answer, [], check

    check["retried"] = True
    messages.append(HumanMessage(content=correction_message(invalid, valid)))
    retry = model.invoke(messages)
    if not retry.tool_calls and retry.text.strip():  # a retry that asks for tools is ignored
        messages.append(retry)
        answer = normalize_answer(retry.text.strip())
        invalid = invalid_tags(answer, valid)
    if not invalid:
        return answer, [FLAG_CITATION_RETRIED], check

    still_invalid = set(invalid)
    check["stripped"] = sum(1 for key in cited_tags(answer) if key in still_invalid)
    return strip_citations(answer, still_invalid), [FLAG_CITATION_RETRIED, FLAG_CITATION_INVALID], check


def nothing_found(tool_calls: list[dict[str, Any]], chunks: list[RetrievedChunk]) -> bool:
    """True for a pure policy question whose searches all came back empty (FR-8 short-circuit)."""
    return bool(tool_calls) and all(c["name"] == SEARCH_POLICY for c in tool_calls) and not chunks


NO_CHECK: dict[str, Any] = {"total": 0, "invalid": 0, "retried": False, "stripped": 0}


def result(
    answer: str,
    tool_calls: list[dict[str, Any]],
    chunks: list[RetrievedChunk],
    flags: list[str] | None = None,
    citation_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The turn's output, answer normalised.

    chunk_ids are every chunk retrieved and flags/citation_check the validation outcome,
    all for the audit log (FR-18) and the Phase 8 eval.
    """
    answer = normalize_answer(answer)
    return {
        "answer": answer,
        "citations": build_citations(answer, chunks),
        "tool_calls": tool_calls,
        "chunk_ids": list(dict.fromkeys(c.chunk_id for c in chunks)),
        "flags": flags or [],
        "citation_check": citation_check or dict(NO_CHECK),
    }


def run_agent(
    ctx: RequestContext,
    message: str,
    history: list[dict[str, str]] | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Answer one user message with tools.

    Returns {answer, citations, tool_calls, chunk_ids, flags, citation_check}.
    `as_of` fixes "today" in both the prompt and the tools (tests, eval).
    Example: run_agent(load_context("u_priya"), "How many options have I vested?", as_of=date(2026, 10, 3))
    -> answer mentions 2,100; tool_calls == [{"name": "get_vesting_status", "args": {}}]
    """
    tools = build_tools(ctx, as_of)
    by_name = {t.name: t for t in tools}
    model = get_chat_model().bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(content=build_system_prompt(ctx, as_of)),
        *to_messages(history),
        HumanMessage(content=message),
    ]
    tool_calls: list[dict[str, Any]] = []
    chunks: list[RetrievedChunk] = []

    for _ in range(MAX_MODEL_CALLS):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            if not reply.text.strip():
                return result(NO_ANSWER, tool_calls, chunks)
            answer, flags, check = validate_citations(model, messages, reply.text, chunks)
            return result(answer, tool_calls, chunks, flags, check)
        for call in reply.tool_calls:
            if len(tool_calls) >= MAX_TOOL_CALLS:
                # Every tool call must get a reply, or the provider rejects the next request.
                messages.append(ToolMessage(LIMIT_REACHED, tool_call_id=call["id"], name=call["name"]))
                continue
            tool_calls.append({"name": call["name"], "args": call["args"]})
            tool_message = run_tool(by_name, call)
            messages.append(tool_message)
            if call["name"] == SEARCH_POLICY and isinstance(tool_message.artifact, list):
                chunks.extend(tool_message.artifact)
        if nothing_found(tool_calls, chunks):
            return result(NOT_FOUND_MESSAGE, tool_calls, chunks)  # no second LLM call

    return result(NO_ANSWER, tool_calls, chunks)
