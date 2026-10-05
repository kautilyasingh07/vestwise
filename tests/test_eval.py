"""Tests for scripts/eval.py's scoring, rate-limit handling and aggregation. No LLM, no Mongo."""

from typing import Any

import pytest

try:
    from scripts import eval as ev
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)


# --- figures in answers ---

@pytest.mark.parametrize(("answer", "value", "found"), [
    ("You have vested 2,100 options.", 2100, True),
    ("You have vested 2100 options.", 2100, True),
    ("You have vested 12,100 options.", 2100, False),      # not a substring match
    ("Next vest: 100 options.", 100, True),
    ("You have vested 2,100 options.", 100, False),         # 100 inside 2,100
    ("Vested so far: 0.", 0, True),                         # sentence-final 0
    ("Strike price 10.5 per share", 10, False),             # 10 inside 10.5
    ("None of your options have vested.", 0, False),        # words don't count as the figure
    ("Next vest on 2026-11-01.", "2026-11-01", True),
    ("Next vest on 1 November 2026.", "2026-11-01", True),
    ("Next vest on November 1, 2026.", "2026-11-01", True),
    ("Next vest on 1 Nov 2026.", "2026-11-01", True),
    ("Next vest on 11 November 2026.", "2026-11-01", False),
    ("Cliff on 15 March 2027", "2027-03-15", True),
])
def test_contains_figure(answer: str, value: Any, found: bool) -> None:
    assert ev.contains_figure(answer, value) is found


# --- rate limits ---

class FakeResponse:
    def __init__(self, retry_after: str | None) -> None:
        self.headers = {"retry-after": retry_after} if retry_after else {}


class Fake429(Exception):
    status_code = 429

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.response = FakeResponse(retry_after)


def test_rate_limit_and_retry_after() -> None:
    exc = Fake429("Rate limit reached ... tokens per minute (TPM)", "7")
    assert ev.is_rate_limit(exc) and ev.retry_after_s(exc) == 7.0
    assert not ev.is_rate_limit(ValueError("bad request"))


@pytest.mark.parametrize(("message", "retry_after", "daily"), [
    ("Rate limit reached ... on tokens per day (TPD): Limit 200000", "926", True),
    ("Rate limit reached ... on requests per day (RPD)", None, True),
    ("Rate limit reached ... on tokens per minute (TPM)", "6", False),
    ("Rate limit reached ... (TPM)", "900", True),  # an unusually long wait is treated as a quota
])
def test_daily_quota_detection(message: str, retry_after: str | None, daily: bool) -> None:
    exc = Fake429(message, retry_after)
    assert ev.is_daily_quota(exc, ev.retry_after_s(exc)) is daily


class ScriptedInner:
    """A chat model that raises the given errors in order, then answers."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)

    def invoke(self, messages: list[Any]) -> Any:
        if self.errors:
            raise self.errors.pop(0)

        class Reply:
            usage_metadata = {"total_tokens": 1234}
        return Reply()


def test_paced_model_retries_429_and_counts_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(ev.time, "sleep", slept.append)
    monkeypatch.setattr(ev.random, "uniform", lambda a, b: 0.0)
    pacer = ev.Pacer()
    model = ev.PacedModel(ScriptedInner([Fake429("TPM", "3"), Fake429("TPM", None)]), pacer)
    model.invoke([])
    assert slept == [3.0, ev.BACKOFF_BASE_S * 2]  # retry-after honoured, then exponential backoff
    stats = pacer.current
    assert (stats.calls, stats.rate_limited, stats.wait_s, stats.tokens) == (1, 2, 3.0 + ev.BACKOFF_BASE_S * 2, 1234)


def test_paced_model_stops_on_daily_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev.time, "sleep", lambda s: pytest.fail("must not sleep through a daily quota"))
    model = ev.PacedModel(ScriptedInner([Fake429("tokens per day (TPD)", "926")]), ev.Pacer())
    with pytest.raises(ev.QuotaExhausted):
        model.invoke([])


def test_non_rate_limit_errors_propagate() -> None:
    model = ev.PacedModel(ScriptedInner([ValueError("boom")]), ev.Pacer())
    with pytest.raises(ValueError):
        model.invoke([])


# --- scoring and aggregation ---

def record(qid: str, kind: str, outcome: str, *, expect_refusal: bool = False, hit: bool | None = None,
           numbers: list[tuple[Any, bool]] = (), cited: bool | None = None, leak: bool | None = None,
           check: dict[str, Any] | None = None, flags: list[str] = (), answer: str = "",
           latency: float = 2.0, wait: float = 0.0) -> dict[str, Any]:
    nums = [{"value": v, "found": f} for v, f in numbers]
    refused = outcome in ev.REFUSAL_OUTCOMES
    return {
        "id": qid, "type": kind, "outcome": outcome, "expect_refusal": expect_refusal, "answer": answer,
        "error": None, "flags": list(flags), "raw_hit_rank": 1 if hit else None,
        "citation_check": check or {"total": 0, "invalid": 0, "retried": False, "stripped": 0},
        "latency_s": latency + wait, "latency_without_waits_s": latency, "wait_s": wait,
        "llm_calls": 2, "rate_limited": 1 if wait else 0, "tokens": 3000,
        "checks": {
            "hit": hit, "numbers": nums,
            "numbers_ok": all(f for _, f in numbers) if kind in ("personal", "mixed") else None,
            "policy_figures_ok": None, "refused": refused,
            "refusal_ok": (refused and not leak) if expect_refusal else None,
            "leak": leak, "cited": cited, "final_invalid_tags": [],
        },
    }


def test_summarize_counts_each_metric() -> None:
    records = [
        record("G05", "policy", "answered", hit=True, cited=True, answer="90 days [ESOP Policy, p. 5].",
               check={"total": 2, "invalid": 1, "retried": True, "stripped": 0}, flags=["citation_retried"]),
        record("G07", "policy", "answered", hit=False, cited=False),
        record("G09", "personal", "answered", numbers=[(2100, True)]),
        record("G14", "mixed", "answered", hit=True, numbers=[(2200, True), (90, False)],
               check={"total": 1, "invalid": 0, "retried": False, "stripped": 0}),
        record("G16", "not_found", "not_found", expect_refusal=True),
        record("G20", "access", "refused", expect_refusal=True, leak=False),
        record("G18", "not_found", "answered", expect_refusal=True, latency=8.0, wait=20.0),
    ]
    s = ev.summarize(records)
    assert (s["hit_at_5_agent"]["ok"], s["hit_at_5_agent"]["of"]) == (2, 3)
    assert (s["number_accuracy"]["ok"], s["number_accuracy"]["of"]) == (1, 2)
    assert (s["number_items_found"]["ok"], s["number_items_found"]["of"]) == (2, 3)
    assert (s["refusal_accuracy"]["ok"], s["refusal_accuracy"]["of"]) == (2, 3)
    assert s["access_leaks"] == 0 and s["false_refusals"] == 0
    assert (s["citation_rate"]["ok"], s["citation_rate"]["of"]) == (1, 2)
    assert s["citation_precision_first_draft"] == {"valid": 2, "of": 3, "rate": 0.667}
    assert s["citation_retries"] == {"retried": 1, "of_answers_with_citations": 2, "stripped_turns": 0}
    assert s["latency_s"]["max"] == 28.0 and s["latency_without_waits_s"]["max"] == 8.0
    assert s["latency_without_waits_s"]["share_le_6s"] == round(6 / 7, 3)


def test_false_refusal_and_leak_are_counted() -> None:
    records = [record("G05", "policy", "refused", hit=False, cited=False),
               record("G20", "access", "answered", expect_refusal=True, leak=True)]
    s = ev.summarize(records)
    assert s["false_refusals"] == 1 and s["access_leaks"] == 1 and s["refusal_accuracy"]["ok"] == 0
