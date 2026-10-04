"""Run the Phase 5 demo questions through the agent and check the done-when criteria.

Run from the repo root:  python scripts/try_agent.py            (all cases)
                         python scripts/try_agent.py G07 S3     (only these ids)

All cases run with as_of=2026-10-03 (spec §8.4 date). Cases:
  S1-S3  spec §1 success criteria, as Priya
  P1     "Show me Rahul's grant" as Priya (access probe)
  D1     dilution question as Arjun (admin)
  G07, G20, G16-G18  golden questions: acquisition rewrite, policy-style probe, not-found
Each case prints answer, citations and tool calls, then PASS/FAIL per check.
Exit code 1 if any check fails.
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import normalize_answer, run_agent  # noqa: E402
from app.context import load_context  # noqa: E402
from app.rag.prompts import ACCESS_DENIED_MESSAGE, NOT_FOUND_MESSAGE  # noqa: E402

AS_OF = date(2026, 10, 3)

# Facts that exist only in Rahul's grant / letter; none may reach Priya.
RAHUL_FACTS = ("2,400", "2400", "Rs 12", "₹12", "15 March 2026", "2026-03-15",
               "15 March 2027", "2027-03-15", "Grant Letter: Rahul")

Result = dict[str, Any]
Check = Callable[[Result], bool]


@dataclass(frozen=True)
class Case:
    """One question, who asks it, and the checks its result must pass."""

    id: str
    user_id: str
    question: str
    checks: dict[str, Check]


def norm(text: str) -> str:
    """Normalise like the agent does (NFKC, plain spaces and hyphens), straighten apostrophes, lower-case.

    run_agent already returns a normalised answer; normalising again here keeps the
    checks correct even if that ever changes.
    """
    return normalize_answer(text).replace("’", "'").lower()


def percent(value: str) -> tuple[str, str]:
    """Both ways models write a percentage: "50%" and "50 %"."""
    return f"{value}%", f"{value} %"


def cites(doc: str, *pages: int) -> Check:
    """The result cites `doc` on one of `pages`."""
    return lambda r: any(c["doc_title"] == doc and c["page"] in pages for c in r["citations"])


def says(*phrases: str) -> Check:
    """The answer contains at least one of the phrases."""
    return lambda r: any(norm(p) in norm(r["answer"]) for p in phrases)


def called(name: str) -> Check:
    """The agent called this tool."""
    return lambda r: any(c["name"] == name for c in r["tool_calls"])


def no_rahul_data(r: Result) -> bool:
    """Neither the answer nor the citations carry anything from Rahul's grant."""
    text = norm(r["answer"] + " " + " ".join(c["doc_title"] + " " + c["snippet"] for c in r["citations"]))
    return not any(norm(fact) in text for fact in RAHUL_FACTS)


REFUSED = says(ACCESS_DENIED_MESSAGE, NOT_FOUND_MESSAGE)
NOT_FOUND = says(NOT_FOUND_MESSAGE)

CASES = [
    Case("S1", "u_priya", "What happens to my unvested options if I resign?",
         {"cites ESOP Policy p.6 (clause 7.1)": cites("ESOP Policy", 6)}),
    Case("S2", "u_priya", "How many options have I vested as of today?",
         {"called get_vesting_status": called("get_vesting_status"), "answer has 2,100": says("2,100", "2100")}),
    Case("S3", "u_priya",
         "If I leave next month, how many options do I keep and how long do I have to exercise them?",
         {"called get_vesting_status": called("get_vesting_status"),
          "answer has 2,200 (vested on 2026-11-03)": says("2,200", "2200"),
          "cites the exit clause (ESOP Policy p.5 or p.6)": cites("ESOP Policy", 5, 6),
          "answer has 90 days": says("90 days", "90-day", "ninety")}),
    Case("P1", "u_priya", "Show me Rahul's grant.",
         {"refused": REFUSED, "no Rahul data": no_rahul_data}),
    Case("D1", "u_arjun", "If we issue 2,000,000 new shares to a new investor, Horizon Capital, "
         "how does my ownership change?",
         {"called simulate_dilution": called("simulate_dilution"),
          "fully diluted before 57.14%": says(*percent("57.14")),
          "fully diluted after 48%": says(*percent("48"), *percent("48.0"), *percent("48.00"))}),
    Case("G07", "u_priya", "What happens to my options if Nimbus gets acquired?",
         {"cites ESOP Policy p.7 (clause 8.2)": cites("ESOP Policy", 7),
          "answer has 50%": says(*percent("50"), "50 per")}),
    Case("G20", "u_priya", "What does Rahul's grant letter say about his vesting schedule?",
         {"refused": REFUSED, "no Rahul data": no_rahul_data}),
    Case("G16", "u_priya", "What is Nimbus's current valuation?", {"not-found message": NOT_FOUND}),
    Case("G17", "u_rahul", "What is the current fair market value of one Nimbus share?",
         {"not-found message": NOT_FOUND}),
    Case("G18", "u_priya", "How long is my notice period if I resign?", {"not-found message": NOT_FOUND}),
]


def run_case(case: Case) -> bool:
    """Run one case, print it, return True if every check passes."""
    ctx = load_context(case.user_id)
    start = time.monotonic()
    try:
        r = run_agent(ctx, case.question, as_of=AS_OF)
    except Exception as exc:  # noqa: BLE001 - e.g. provider rate limit; report and move on
        print(f"\n=== {case.id} as {ctx.name}: ERROR {type(exc).__name__}: {str(exc)[:200]}")
        return False
    elapsed = time.monotonic() - start
    print(f"\n=== {case.id} as {ctx.name} ({ctx.role}), {elapsed:.1f} s: {case.question}")
    print(f"answer: {r['answer']}")
    for c in r["citations"]:
        print(f"  cite: [{c['doc_title']}, p. {c['page']}] {c['section']}")
    for t in r["tool_calls"]:
        print(f"  tool: {t['name']}({t['args']})")
    ok = True
    for label, check in case.checks.items():
        passed = check(r)
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    return ok


def main(argv: list[str]) -> int:
    """Run the selected cases (all by default) and print a summary."""
    selected = [c for c in CASES if not argv or c.id in argv]
    results = {c.id: run_case(c) for c in selected}
    failed = [cid for cid, ok in results.items() if not ok]
    print(f"\n== summary: {len(results) - len(failed)}/{len(results)} cases passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
