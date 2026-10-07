"""Live follow-up check (FR-16): two chat turns as Priya, the second relying on history.

Run from the repo root:  python scripts/try_followup.py

  turn 1: "How many options have I vested?"
  turn 2: "And in March next year?"   (history = turn 1's question and answer)

Pass when turn 2 calls get_vesting_status with a March 2027 date and its answer
states the figure the tool returns for that date (not turn 1's figure).

Free-tier budget (Groq gpt-oss-120b, 8K tokens/min): every model call, including
any citation retry, goes through PacedModel, which allows at most MAX_CALLS calls
spaced MIN_GAP_S apart and raises instead of making a 5th. The provider client's
own retries are off (max_retries=0), so a 429 can't add hidden calls.
Expected figures come from invoking the tool directly (Mongo only, no LLM).
"""

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import BaseMessage  # noqa: E402

from app import agent  # noqa: E402
from app.agent import normalize_answer, run_agent  # noqa: E402
from app.context import load_context  # noqa: E402
from app.llm import get_chat_model  # noqa: E402
from app.tools.factory import build_tools  # noqa: E402

AS_OF = date(2026, 10, 3)  # project demo date: "next year" = 2027
USER = "u_priya"
TURN_1 = "How many options have I vested?"
TURN_2 = "And in March next year?"
MAX_CALLS = 4
MIN_GAP_S = 30.0


class BudgetExceeded(RuntimeError):
    """Raised instead of making a model call beyond MAX_CALLS."""


class PacedModel:
    """Wraps the real chat model: counts calls, enforces the budget and the gap between calls."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.bound = inner, inner
        self.calls = 0
        self.last_start: float | None = None

    def bind_tools(self, tools: list[Any]) -> "PacedModel":
        self.bound = self.inner.bind_tools(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> Any:
        if self.calls >= MAX_CALLS:
            raise BudgetExceeded(f"a call beyond the budget of {MAX_CALLS} was refused")
        if self.last_start is not None:
            wait = MIN_GAP_S - (time.monotonic() - self.last_start)
            if wait > 0:
                print(f"    (pacing: waiting {wait:.0f} s)")
                time.sleep(wait)
        self.calls += 1
        self.last_start = time.monotonic()
        print(f"    model call {self.calls}/{MAX_CALLS}")
        return self.bound.invoke(messages)


def vested_on(as_of: str | None) -> int:
    """Vested options per the tool itself (deterministic; no LLM)."""
    tools = {t.name: t for t in build_tools(load_context(USER), AS_OF)}
    out = json.loads(tools["get_vesting_status"].invoke({"as_of": as_of} if as_of else {}))
    return sum(g["vested"] for g in out["grants"])


def says(answer: str, n: int) -> bool:
    """Does the normalised answer contain n as "2,600" or "2600"?"""
    text = normalize_answer(answer)
    return f"{n:,}" in text or str(n) in text


def check(label: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return ok


def main() -> int:
    """Run both turns and print the checks; exit 1 on any failure."""
    paced = PacedModel(get_chat_model(max_retries=0))
    agent.get_chat_model = lambda: paced  # run_agent builds its model through this name
    ctx = load_context(USER)

    print(f"turn 1 as {ctx.name}: {TURN_1}")
    first = run_agent(ctx, TURN_1, as_of=AS_OF)
    print(f"  answer: {first['answer']}\n  tool calls: {first['tool_calls']}")
    history = [{"role": "user", "content": TURN_1}, {"role": "assistant", "content": first["answer"]}]

    print(f"\nturn 2: {TURN_2}   (history: {len(history)} messages)")
    second = run_agent(ctx, TURN_2, history=history, as_of=AS_OF)
    print(f"  answer: {second['answer']}\n  tool calls: {second['tool_calls']}")

    today_n = vested_on(None)
    vest_calls = [c for c in second["tool_calls"] if c["name"] == "get_vesting_status"]
    asked = [str(c["args"].get("as_of") or "") for c in vest_calls]
    march = next((d for d in asked if d.startswith("2027-03")), None)
    march_n = vested_on(march) if march else None
    print(f"\nexpected from the tool: {today_n:,} on {AS_OF}; {march_n:,} on {march}" if march_n is not None
          else f"\nexpected from the tool: {today_n:,} on {AS_OF}")

    results = [
        check("turn 1 called get_vesting_status", any(c["name"] == "get_vesting_status" for c in first["tool_calls"])),
        check(f"turn 1 states {today_n:,}", says(first["answer"], today_n)),
        check(f"turn 2 called get_vesting_status with a March 2027 date (asked {asked})", march is not None),
        check(f"turn 2 states the tool's March figure ({march_n:,})" if march_n else "turn 2 states the March figure",
              march_n is not None and says(second["answer"], march_n)),
        check(f"turn 2 does not repeat turn 1's figure ({today_n:,})",
              march_n == today_n or not says(second["answer"], today_n)),
    ]
    print(f"\nmodel calls used: {paced.calls}/{MAX_CALLS}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BudgetExceeded as exc:
        print(f"STOPPED: {exc}")
        sys.exit(1)
