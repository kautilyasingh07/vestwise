"""Tool-calling agent: one chat turn (spec §6 request flow, §8.6, FR-14, FR-15).

The loop, written out so every step is visible:
  1. send system prompt + history + question to the model, with the user's tools bound;
  2. if the reply contains tool calls, run them (our code, with ctx captured in the tools)
     and append each result as a ToolMessage;
  3. repeat until the model answers in plain text, or the cap of MAX_TOOL_CALLS is hit.

Citations are built from the chunks search_policy actually returned, keeping only the
ones whose [title, p. N] tag appears in the answer. A tag the model made up matches no
retrieved chunk, so it never becomes a citation.
"""

import re
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
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
TAG_RE = re.compile(r"^(.+?),\s*p\.\s*(\d+)$")  # "ESOP Policy, p. 6"

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


def cited_tags(answer: str) -> list[tuple[str, int]]:
    """(title, page) for every tag in the answer, in order.

    Handles single tags "[ESOP Policy, p. 6]" and combined ones the model sometimes
    writes, "[ESOP Policy, p. 6; Grant Letter: Priya Sharma, p. 2]". Parts that
    aren't "title, p. N" (e.g. "[get_vesting_status]") are ignored.
    """
    tags = []
    for group in BRACKET_RE.findall(answer):
        for part in group.split(";"):
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


def nothing_found(tool_calls: list[dict[str, Any]], chunks: list[RetrievedChunk]) -> bool:
    """True for a pure policy question whose searches all came back empty (FR-8 short-circuit)."""
    return bool(tool_calls) and all(c["name"] == SEARCH_POLICY for c in tool_calls) and not chunks


def result(answer: str, tool_calls: list[dict[str, Any]], chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """The turn's output; chunk_ids are every chunk retrieved, for the audit log (FR-18)."""
    return {
        "answer": answer,
        "citations": build_citations(answer, chunks),
        "tool_calls": tool_calls,
        "chunk_ids": list(dict.fromkeys(c.chunk_id for c in chunks)),
    }


def run_agent(
    ctx: RequestContext,
    message: str,
    history: list[dict[str, str]] | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Answer one user message with tools; returns {answer, citations, tool_calls, chunk_ids}.

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
            return result(reply.text.strip() or NO_ANSWER, tool_calls, chunks)
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
