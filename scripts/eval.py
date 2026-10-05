"""End-to-end evaluation on the golden set (spec §12, §5 targets).

Run from the repo root:
  python scripts/eval.py --label baseline            # all 20 questions, writes eval/results-<UTC>.json
  python scripts/eval.py --only G07,G13 --label probe # a subset
  python scripts/eval.py --compare eval/results-A.json eval/results-B.json   # run-to-run variance

Every golden question goes through the real agent (run_agent) as its user, with its
`as_of`, against the real model (LLM_PROVIDER / LLM_MODEL) and Atlas. Metrics:

  retrieval hit@5      policy + mixed: the expected (doc, page) is among the chunks the agent's
                       own search_policy calls returned (each call returns <= 5). The Phase 4
                       raw-question hit@5 is reported alongside, for comparison.
  number accuracy      personal + mixed: every expected number/date appears in the answer.
  refusal accuracy     not_found + access: outcome is refused/not_found, and access probes leak nothing.
  false refusals       answerable questions that were refused.
  citation rate        policy questions whose answer has at least one citation (spec §5 grounding).
  citation precision   first draft: valid in-text citations / all in-text citations (before the
                       corrective retry), plus how often the retry fired and how often tags were stripped.
  latency              per question, with and without rate-limit waits.

Rate limits: the model is built with its client retries off; this script retries HTTP 429
itself (honouring retry-after, else exponential backoff), so time spent waiting is measured
separately from time spent answering.
"""

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent as agent_module  # noqa: E402
from app import db  # noqa: E402
from app.agent import cited_tags, run_agent  # noqa: E402
from app.audit import classify  # noqa: E402
from app.config import settings  # noqa: E402
from app.context import RequestContext, load_context  # noqa: E402
from app.llm import get_chat_model  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from scripts.eval_retrieval import GOLDEN_PATH, hit_rank, load_golden  # noqa: E402
from scripts.try_agent import RAHUL_FACTS, norm  # noqa: E402

RESULTS_DIR = ROOT / "eval"
MAX_ATTEMPTS = 8  # per LLM call, on HTTP 429
MAX_WAIT_S = 120.0  # a 429 asking for a longer wait is a quota, not a burst: stop instead of sleeping
BACKOFF_BASE_S = 4.0
BACKOFF_CAP_S = 60.0
NFR_LATENCY_S = 6.0  # spec §5
REFUSAL_OUTCOMES = {"refused", "not_found"}
TARGET_RATE = 18 / 20  # spec §5 hit@5, applied as a rate (Phase 4)


# --- rate-limit aware model wrapper ---

@dataclass
class CallStats:
    """Per-question accounting of LLM calls."""

    calls: int = 0
    rate_limited: int = 0
    wait_s: float = 0.0
    tokens: int = 0  # total (input + output) tokens reported by the provider


class QuotaExhausted(RuntimeError):
    """A 429 for a daily quota (e.g. Groq's tokens-per-day) or with a very long retry-after."""


@dataclass
class Pacer:
    """Holds the stats object for the question currently running."""

    current: CallStats = field(default_factory=CallStats)


def is_rate_limit(exc: Exception) -> bool:
    """True for HTTP 429 from any provider SDK (Groq raises RateLimitError with status_code 429)."""
    return getattr(exc, "status_code", None) == 429 or "RateLimit" in type(exc).__name__ or " 429" in str(exc)[:300]


def is_daily_quota(exc: Exception, delay: float | None) -> bool:
    """Groq names the limit in the message: '(TPD)' tokens per day, '(RPD)' requests per day."""
    text = str(exc)
    return "per day" in text or "(TPD)" in text or "(RPD)" in text or (delay is not None and delay > MAX_WAIT_S)


def retry_after_s(exc: Exception) -> float | None:
    """Seconds from the 429 response's retry-after header, if present."""
    response = getattr(exc, "response", None)
    value = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class PacedModel:
    """Stands in for the chat model inside run_agent: retries 429 with backoff and times the waits."""

    def __init__(self, inner: Any, pacer: Pacer) -> None:
        self.inner, self.pacer = inner, pacer

    def bind_tools(self, tools: list[Any]) -> "PacedModel":
        return PacedModel(self.inner.bind_tools(tools), self.pacer)

    def invoke(self, messages: list[Any]) -> Any:
        stats = self.pacer.current
        for attempt in range(MAX_ATTEMPTS):
            try:
                reply = self.inner.invoke(messages)
                stats.calls += 1
                stats.tokens += (getattr(reply, "usage_metadata", None) or {}).get("total_tokens", 0)
                return reply
            except Exception as exc:  # noqa: BLE001 - only 429 is retried; everything else propagates
                if not is_rate_limit(exc) or attempt == MAX_ATTEMPTS - 1:
                    raise
                header = retry_after_s(exc)
                if is_daily_quota(exc, header):
                    raise QuotaExhausted(str(exc)[:400]) from exc
                delay = header or min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt)
                delay += random.uniform(0, 1)  # jitter
                stats.rate_limited += 1
                stats.wait_s += delay
                time.sleep(delay)
        raise RuntimeError("unreachable")


def install_paced_model(pacer: Pacer) -> None:
    """Make run_agent use a 429-retrying wrapper around a client whose own retries are off."""
    agent_module.get_chat_model = lambda: PacedModel(get_chat_model(max_retries=0), pacer)


# --- scoring ---

def number_forms(value: int | str) -> list[str]:
    """Equivalent written forms of an expected figure: 2100 -> "2,100", "2100"; a date -> ISO and long forms."""
    if isinstance(value, int):
        return list(dict.fromkeys([f"{value:,}", str(value)]))
    d = date.fromisoformat(value)
    return [d.isoformat(), f"{d.day} {d:%B} {d.year}", f"{d:%B} {d.day}, {d.year}",
            f"{d.day} {d:%b} {d.year}", f"{d:%b} {d.day}, {d.year}"]


def contains_figure(answer: str, value: int | str) -> bool:
    """The figure appears as a whole number/date (so 100 doesn't match inside 2,100 or 1000)."""
    text = norm(answer)
    for form in number_forms(value):
        pattern = r"(?<![\d.,])" + re.escape(form.lower()) + r"(?!\d|[.,]\d)"
        if re.search(pattern, text):
            return True
    return False


def retrieved_pages(chunk_ids: list[str]) -> set[tuple[str, int]]:
    """(doc_title, page) of the chunks the agent's searches returned."""
    if not chunk_ids:
        return set()
    return {(c["doc_title"], c["page"]) for c in db.chunks().find({"_id": {"$in": chunk_ids}}, {"doc_title": 1, "page": 1})}


def raw_hit(row: dict[str, Any], ctx: RequestContext) -> int | None:
    """Phase 4 metric: rank of the expected page when the raw question is the query (no LLM)."""
    chunks = retrieve(row["question"], ctx.company_id, ctx.role, ctx.stakeholder_id, k=5, min_score=-1.0)
    return hit_rank(chunks, row["expected_doc"], row["expected_page"])


def leaks(answer: str, citations: list[dict[str, Any]]) -> bool:
    """An access probe's answer or citations contain anything from Rahul's grant."""
    text = norm(answer + " " + " ".join(c["doc_title"] + " " + c["snippet"] for c in citations))
    return any(norm(fact) in text for fact in RAHUL_FACTS)


def score(row: dict[str, Any], out: dict[str, Any], pages: set[tuple[str, int]]) -> dict[str, Any]:
    """Per-question checks; None where a metric doesn't apply to this question type."""
    kind, answer = row["type"], out["answer"]
    numbers = [{"value": v, "found": contains_figure(answer, v)} for v in row["expected_numbers"]]
    checks: dict[str, Any] = {
        "hit": (row["expected_doc"], row["expected_page"]) in pages if kind in ("policy", "mixed") else None,
        "numbers": numbers,
        "numbers_ok": all(n["found"] for n in numbers) if kind in ("personal", "mixed") else None,
        "policy_figures_ok": (all(n["found"] for n in numbers) if numbers else None) if kind == "policy" else None,
        "refused": out["outcome"] in REFUSAL_OUTCOMES,
        "refusal_ok": None,
        "leak": None,
        "cited": bool(out["citations"]) if kind == "policy" else None,
    }
    if row["expect_refusal"]:
        checks["leak"] = leaks(answer, out["citations"]) if kind == "access" else None
        checks["refusal_ok"] = checks["refused"] and not checks["leak"]
    final_tags = cited_tags(answer)
    checks["final_invalid_tags"] = [list(t) for t in final_tags if t not in pages]
    return checks


# --- one run ---

def run_question(row: dict[str, Any], ctx: RequestContext, pacer: Pacer) -> dict[str, Any]:
    """Run one golden question through the agent and score it."""
    pacer.current = stats = CallStats()
    as_of = date.fromisoformat(row["as_of"])
    start = time.monotonic()
    error = None
    try:
        out = run_agent(ctx, row["question"], None, as_of)
    except QuotaExhausted:
        raise  # stops the whole run (see run())
    except Exception as exc:  # noqa: BLE001 - a failed question is a result, not a crash
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        out = {"answer": "", "citations": [], "tool_calls": [], "chunk_ids": [], "flags": [],
               "citation_check": {"total": 0, "invalid": 0, "retried": False, "stripped": 0}}
    wall = time.monotonic() - start
    out["outcome"] = classify(out["answer"] if not error else None, error)
    pages = retrieved_pages(out["chunk_ids"])
    record = {
        "id": row["id"], "type": row["type"], "user_id": row["user_id"], "question": row["question"],
        "as_of": row["as_of"], "expected_doc": row["expected_doc"], "expected_page": row["expected_page"],
        "expected_numbers": row["expected_numbers"], "expect_refusal": row["expect_refusal"],
        "answer": out["answer"], "outcome": out["outcome"], "error": error,
        "citations": [{k: c[k] for k in ("doc_title", "page", "section")} for c in out["citations"]],
        "tool_calls": out["tool_calls"], "retrieved_pages": sorted([list(p) for p in pages]),
        "flags": out["flags"], "citation_check": out["citation_check"],
        "raw_hit_rank": raw_hit(row, ctx) if row["type"] in ("policy", "mixed") else None,
        "latency_s": round(wall, 2), "latency_without_waits_s": round(wall - stats.wait_s, 2),
        "llm_calls": stats.calls, "rate_limited": stats.rate_limited, "wait_s": round(stats.wait_s, 2),
        "tokens": stats.tokens,
    }
    record["checks"] = score(row, out, pages)
    return record


def rate(values: list[bool]) -> dict[str, Any]:
    """{"ok": n, "of": m, "rate": n/m}."""
    ok = sum(1 for v in values if v)
    return {"ok": ok, "of": len(values), "rate": round(ok / len(values), 3) if values else None}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics over all questions of a run."""
    def of(kinds: tuple[str, ...], key: str) -> list[bool]:
        return [r["checks"][key] for r in records if r["type"] in kinds and r["checks"][key] is not None]

    drafts = [r["citation_check"] for r in records]
    total, invalid = sum(d["total"] for d in drafts), sum(d["invalid"] for d in drafts)
    final_total = sum(len(cited_tags(r["answer"])) for r in records)
    final_invalid = sum(len(r["checks"]["final_invalid_tags"]) for r in records)
    with_tags = [d for d in drafts if d["total"]]
    lat = [r["latency_s"] for r in records]
    lat_nw = [r["latency_without_waits_s"] for r in records]

    def dist(xs: list[float]) -> dict[str, float]:
        xs = sorted(xs)
        return {"median": round(statistics.median(xs), 2), "p90": round(xs[math.ceil(0.9 * len(xs)) - 1], 2),  # nearest rank
                "max": round(xs[-1], 2), "share_le_6s": round(sum(x <= NFR_LATENCY_S for x in xs) / len(xs), 3)}

    raw = [r["raw_hit_rank"] is not None for r in records if r["type"] in ("policy", "mixed")]
    return {
        "questions": len(records),
        "errors": sum(1 for r in records if r["error"]),
        "hit_at_5_agent": rate(of(("policy", "mixed"), "hit")),
        "hit_at_5_raw_question": rate(raw),
        "number_accuracy": rate(of(("personal", "mixed"), "numbers_ok")),
        "number_items_found": rate([n["found"] for r in records if r["type"] in ("personal", "mixed")
                                    for n in r["checks"]["numbers"]]),
        "policy_figure_accuracy": rate(of(("policy",), "policy_figures_ok")),
        "refusal_accuracy": rate(of(("not_found", "access"), "refusal_ok")),
        "access_leaks": sum(1 for r in records if r["checks"]["leak"]),
        "false_refusals": sum(1 for r in records if not r["expect_refusal"] and r["checks"]["refused"]),
        "citation_rate": rate(of(("policy",), "cited")),
        "citation_precision_first_draft": {"valid": total - invalid, "of": total,
                                           "rate": round((total - invalid) / total, 3) if total else None},
        "citation_precision_final": {"valid": final_total - final_invalid, "of": final_total,
                                     "rate": round((final_total - final_invalid) / final_total, 3) if final_total else None},
        "citation_retries": {"retried": sum(1 for d in with_tags if d["retried"]), "of_answers_with_citations": len(with_tags),
                             "stripped_turns": sum(1 for r in records if "citation_invalid" in r["flags"])},
        "latency_s": dist(lat),
        "latency_without_waits_s": dist(lat_nw),
        "llm_calls": sum(r["llm_calls"] for r in records),
        "tokens": sum(r["tokens"] for r in records),
        "rate_limited_calls": sum(r["rate_limited"] for r in records),
        "wait_s_total": round(sum(r["wait_s"] for r in records), 1),
    }


TARGETS: list[tuple[str, str, Any]] = [
    # (label, summary path, check) -- spec §5 / §12
    ("Retrieval hit@5 (agent)", "hit_at_5_agent", lambda m: m["rate"] >= TARGET_RATE),
    ("Retrieval hit@5 (raw question)", "hit_at_5_raw_question", lambda m: m["rate"] >= TARGET_RATE),
    ("Number accuracy", "number_accuracy", lambda m: m["rate"] == 1.0),
    ("Refusal accuracy", "refusal_accuracy", lambda m: m["rate"] == 1.0),
    ("Access leaks", "access_leaks", lambda m: m == 0),
    ("False refusals", "false_refusals", lambda m: m == 0),
    ("Citation rate (policy)", "citation_rate", lambda m: m["rate"] == 1.0),
    ("Citation precision, first draft", "citation_precision_first_draft", None),
    ("Citation precision, final", "citation_precision_final", lambda m: m["rate"] in (None, 1.0)),
    ("Policy figure accuracy (extra)", "policy_figure_accuracy", None),
]
TARGET_TEXT = {
    "hit_at_5_agent": ">= 90% (18/20 rate)", "hit_at_5_raw_question": ">= 90%", "number_accuracy": "100%",
    "refusal_accuracy": "100%", "access_leaks": "0", "false_refusals": "0", "citation_rate": "100%",
    "citation_precision_first_draft": "report", "citation_precision_final": "100%", "policy_figure_accuracy": "report",
}


def fmt(metric: Any) -> str:
    """'10/11 (90.9%)' for rates, the number otherwise."""
    if isinstance(metric, dict) and "rate" in metric:
        num = metric.get("ok", metric.get("valid"))
        return f"{num}/{metric['of']} ({metric['rate']:.1%})" if metric["rate"] is not None else "n/a"
    return str(metric)


def print_summary(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Per-question line, then the metrics table against the spec targets."""
    print("\n== per question ==")
    for r in records:
        c = r["checks"]
        marks = []
        for key, label in (("hit", "hit"), ("numbers_ok", "nums"), ("refusal_ok", "refusal"), ("cited", "cited")):
            if c[key] is not None:
                marks.append(f"{label}:{'ok' if c[key] else 'FAIL'}")
        if c["numbers"] and not c["numbers_ok"] and c["numbers_ok"] is not None:
            marks.append("missing " + ",".join(str(n["value"]) for n in c["numbers"] if not n["found"]))
        if r["flags"]:
            marks.append("flags " + ",".join(r["flags"]))
        if r["error"]:
            marks.append("ERROR " + r["error"][:60])
        print(f"  {r['id']} {r['type']:<9} {r['outcome']:<9} {r['latency_without_waits_s']:5.1f}s "
              f"(+{r['wait_s']:.0f}s wait) {' '.join(marks)}")

    print("\n== summary ==")
    print(f"  {'metric':<34} {'value':<22} {'target':<20} met")
    for label, key, check in TARGETS:
        value = summary[key]
        not_applicable = isinstance(value, dict) and value.get("rate") is None  # e.g. a subset without that type
        met = "-" if check is None or not_applicable else ("yes" if check(value) else "NO")
        print(f"  {label:<34} {fmt(value):<22} {TARGET_TEXT[key]:<20} {met}")
    rt = summary["citation_retries"]
    print(f"  {'Citation retry fired':<34} {rt['retried']}/{rt['of_answers_with_citations']} answers with citations; "
          f"stripped in {rt['stripped_turns']}")
    for key, label in (("latency_s", "Latency incl. rate-limit waits"), ("latency_without_waits_s", "Latency excl. waits")):
        d = summary[key]
        print(f"  {label:<34} median {d['median']}s, p90 {d['p90']}s, max {d['max']}s, <=6s {d['share_le_6s']:.0%}")
    print(f"  LLM calls {summary['llm_calls']}, tokens {summary['tokens']:,}, rate-limited {summary['rate_limited_calls']}, "
          f"waited {summary['wait_s_total']}s; errors {summary['errors']}")


def git_commit() -> str | None:
    """Current commit hash, so a result file can be tied to the code that produced it."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_results(path: Path, header: dict[str, Any], records: list[dict[str, Any]], status: str) -> None:
    """(Re)write the results file; called after every question so a stopped run keeps its data."""
    summary = summarize(records) if records else {}
    path.write_text(json.dumps({**header, "status": status, "completed": len(records), "summary": summary,
                                "questions": records}, indent=2, default=str), encoding="utf-8")


def run(label: str, only: set[str] | None) -> Path:
    """Run the golden set, print the summary, write the JSON results file; return its path.

    Stops early (status "stopped: quota") if the provider's daily quota is exhausted; the
    questions completed so far are kept.
    """
    db.ping()
    golden = load_golden(GOLDEN_PATH)
    rows = [r for r in golden if not only or r["id"] in only]
    pacer = Pacer()
    install_paced_model(pacer)
    contexts = {uid: load_context(uid) for uid in {r["user_id"] for r in rows}}
    started = datetime.now(UTC)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"results-{started:%Y%m%dT%H%M%SZ}.json"
    header = {
        "label": label, "started_at": started.isoformat(), "git_commit": git_commit(),
        "model": f"{settings.llm_provider}/{settings.llm_model}",
        "config": {"retrieval_min_score": settings.retrieval_min_score, "retrieval_top_k": settings.retrieval_top_k,
                   "subset": sorted(only) if only else None},
    }
    print(f"eval '{label}': {len(rows)} questions, model {header['model']}, "
          f"min_score {settings.retrieval_min_score}, k {settings.retrieval_top_k}")
    records: list[dict[str, Any]] = []
    status = "complete"
    for row in rows:
        try:
            record = run_question(row, contexts[row["user_id"]], pacer)
        except QuotaExhausted as exc:
            status = "stopped: quota"
            print(f"  {row['id']} STOPPED: provider quota exhausted. {exc}", flush=True)
            break
        records.append(record)
        write_results(path, header, records, "running")
        print(f"  {record['id']} done: {record['outcome']}, {record['latency_s']}s "
              f"({record['llm_calls']} calls, {record['tokens']:,} tokens, {record['rate_limited']} rate-limited)",
              flush=True)
    write_results(path, header, records, status)
    if records:
        print_summary(summarize(records), records)
    print(f"\nstatus: {status} ({len(records)}/{len(rows)} questions); wrote {path.relative_to(ROOT)}")
    return path


def compare(paths: list[Path]) -> None:
    """Run-to-run variance: metric values side by side, and questions whose verdict changed."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    print("== metrics ==")
    print(f"  {'metric':<34} " + " ".join(f"{r['label'][:18]:<20}" for r in runs))
    for label, key, _ in TARGETS:
        print(f"  {label:<34} " + " ".join(f"{fmt(r['summary'][key]):<20}" for r in runs))
    for key in ("latency_without_waits_s", "latency_s"):
        print(f"  {key + ' median':<34} " + " ".join(f"{r['summary'][key]['median']:<20}" for r in runs))

    def verdict(q: dict[str, Any]) -> str:
        c = q["checks"]
        parts = [f"{k}={'ok' if c[k] else 'FAIL'}" for k in ("hit", "numbers_ok", "refusal_ok", "cited") if c[k] is not None]
        return f"{q['outcome']} " + " ".join(parts) + (" +retry" if q["citation_check"]["retried"] else "")

    print("\n== questions whose verdict differs between runs ==")
    by_id = [{q["id"]: q for q in r["questions"]} for r in runs]
    changed = 0
    for qid in by_id[0]:
        verdicts = [verdict(b[qid]) for b in by_id if qid in b]
        if len(set(verdicts)) > 1:
            changed += 1
            print(f"  {qid}: " + " | ".join(verdicts))
    print(f"  {changed} of {len(by_id[0])} questions changed verdict")
    same_text = sum(1 for qid in by_id[0] if len({b[qid]['answer'] for b in by_id if qid in b}) == 1)
    print(f"  identical answer text across runs: {same_text} of {len(by_id[0])}")


def main(argv: list[str]) -> int:
    """CLI: run the eval, or compare result files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", default="run", help="name for this run, stored in the results file")
    parser.add_argument("--only", help="comma-separated golden ids, e.g. G07,G13")
    parser.add_argument("--compare", nargs="+", type=Path, help="results files to compare instead of running")
    args = parser.parse_args(argv)
    if args.compare:
        compare(args.compare)
        return 0
    run(args.label, set(args.only.split(",")) if args.only else None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
