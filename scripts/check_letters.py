"""Run the compliance checker on the 3 test letters and score it on the planted issues (spec §8.7).

Run from the repo root:
  python scripts/check_letters.py            # uses cached LLM outputs; makes live calls only for what isn't cached
  python scripts/check_letters.py --plan     # only print how many live LLM calls a run would make
  python scripts/check_letters.py --refresh  # ignore the cache (2 live calls per letter)

Rules come from the policy_rules collection (reviewed, loaded with build_policy_rules.py --load);
the pool and board resolution date from Mongo, as the API does. Prints each letter's findings
and report, then precision and recall over the 4 planted issues:

  precision = planted issues flagged / all issues flagged (any non-match finding)
  recall    = planted issues flagged / 4

A planted issue counts only if its finding has the expected status *and* cites both the
letter (page) and the policy. Stops at the first HTTP 429; never retries.
"""

import argparse
import sys
from pathlib import Path

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.compliance.cache import is_quota_error  # noqa: E402
from app.compliance.check import Finding  # noqa: E402
from app.compliance.pipeline import CheckResult, check_letter, planned_calls  # noqa: E402
from app.compliance.rules import get_rules  # noqa: E402
from app.tools import repo  # noqa: E402
from app.tools.captable import pool_status  # noqa: E402

LETTERS_DIR = ROOT / "data" / "docs" / "compliance"
LETTERS = ["grant_letter_ananya.pdf", "grant_letter_vikram.pdf", "grant_letter_neha.pdf"]
COMPANY = "nimbus"

# Spec §8.7 test letters: (file, field, expected status). Ananya is clean.
PLANTED: frozenset[tuple[str, str, str]] = frozenset({
    ("grant_letter_vikram.pdf", "cliff_months", "conflict"),
    ("grant_letter_vikram.pdf", "exercise_window_days", "conflict"),
    ("grant_letter_neha.pdf", "exercise_window_days", "missing"),
    ("grant_letter_neha.pdf", "options", "exceeds_pool"),
})


def fully_cited(finding: Finding) -> bool:
    """Both sources present: the letter (with a page, unless the letter is silent) and at least one policy clause."""
    letter_ok = finding.letter_citation.page is not None or finding.status == "missing"
    return letter_ok and bool(finding.policy_citations)


def flagged(file_name: str, findings: list[Finding]) -> set[tuple[str, str, str]]:
    """Every non-match finding of one letter as (file, field, status)."""
    return {(file_name, f.field, f.status) for f in findings if f.status != "match"}


def score(flags: set[tuple[str, str, str]], hits: set[tuple[str, str, str]],
          planted: frozenset[tuple[str, str, str]] = PLANTED) -> dict[str, float | int]:
    """Precision and recall over the planted issues. `hits` = flags that are planted and fully cited."""
    true_pos = len(hits & planted)
    return {"flagged": len(flags), "planted": len(planted), "true_positives": true_pos,
            "precision": true_pos / len(flags) if flags else 1.0,
            "recall": true_pos / len(planted) if planted else 1.0}


def print_result(result: CheckResult) -> None:
    """Findings table and report of one letter."""
    print(f"\n=== {result.letter_title} ({result.file_name}) -> {result.outcome}")
    print(f"    live LLM calls {result.llm_calls}, cached {result.cached}")
    for f in result.findings:
        policy = " ".join(c.label for c in f.policy_citations) or "-"
        print(f"  {f.id:<4} {f.field:<42} {f.status:<13} letter={f.letter_text:<22} "
              f"{f.letter_citation.label} {policy}")
    for warning in result.warnings:
        print(f"  ! {warning}")
    print(f"  Report ({result.report.source}):")
    for line in result.report.lines:
        print(f"    {line.text}")
    for problem in result.report.problems:
        print(f"    (report check: {problem})")


def main() -> int:
    """Check every test letter, print findings, then precision/recall."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="ignore cached LLM outputs")
    parser.add_argument("--plan", action="store_true", help="only print how many live LLM calls a run would make")
    args = parser.parse_args()

    rules = get_rules(COMPANY)
    if not rules:
        print("No policy rules in Mongo. Review data/policy_rules.json, then run build_policy_rules.py --load.")
        return 1
    pool = pool_status(repo.get_grants(COMPANY), repo.get_pool_size(COMPANY))["unallocated"]
    board_date = repo.get_board_resolution_date(COMPANY)
    paths = [LETTERS_DIR / name for name in LETTERS]
    plan = {p.name: planned_calls(p, rules, pool, board_date, refresh=args.refresh) for p in paths}
    print(f"{len(rules)} rules, pool remaining {pool:,}, board resolution {board_date}")
    print(f"Planned live LLM calls: {sum(plan.values())} {plan}")
    if args.plan:
        return 0

    flags: set[tuple[str, str, str]] = set()
    hits: set[tuple[str, str, str]] = set()
    total_calls = 0
    for path in paths:
        try:
            result = check_letter(path, rules, pool, board_date, refresh=args.refresh)
        except Exception as exc:  # noqa: BLE001 - any LLM failure stops the batch; never retry (budget)
            kind = "quota / rate limit (HTTP 429)" if is_quota_error(exc) else type(exc).__name__
            print(f"\nExtraction failed on {path.name}: {kind}. Stopping, no retries.\n{str(exc)[:300]}")
            return 2
        total_calls += result.llm_calls
        print_result(result)
        flags |= flagged(path.name, result.findings)
        hits |= {(path.name, f.field, f.status) for f in result.findings if f.status != "match" and fully_cited(f)}
        if result.quota_hit:
            print("\nLLM quota / rate limit hit on the report call. Stopping, no retries.")
            return 2

    s = score(flags, hits)
    print(f"\nLive LLM calls this run: {total_calls}")
    print(f"Planted issues flagged with both citations: {s['true_positives']}/{s['planted']}; "
          f"issues flagged in total: {s['flagged']}")
    print(f"Precision {s['precision']:.2f}  Recall {s['recall']:.2f}")
    for missed in sorted(PLANTED - hits):
        print(f"  MISSED  {missed}")
    for extra in sorted(flags - PLANTED):
        print(f"  EXTRA   {extra}")
    return 0 if s["precision"] == 1.0 and s["recall"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
