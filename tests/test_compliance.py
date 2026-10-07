"""Compliance checker tests (spec §8.7, FR-21..25). No test calls the LLM or Mongo.

- compare(): hand-built terms against hand-built rules, every status and boundary.
- schema, rules gate, grounding, report validation and fallback.
- pipeline + cache: fake LLMs count calls; a cached run builds no model at all.
- the API endpoint and audit record, on fakes.
- integration: the 3 test letters, replayed from recorded (cached) real extractions,
  checked against the reviewed data/policy_rules.json, give exactly the spec §8.7 findings.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.compliance import cache as cache_module
from app.compliance.check import POOL_NOTE, Finding, compare, count_statuses
from app.compliance.extract import extract_terms, ground_terms, letter_prompt
from app.compliance.report import build_report, facts_line, template_report, validate_report
from app.compliance.rules import RULES_PATH, load_rule_file, require_loadable
from app.compliance.schema import (
    DateTerm,
    FrequencyTerm,
    GrantTerms,
    IntTerm,
    LeaverTerm,
    LeaverTerms,
    NumberTerm,
    PolicyRule,
    RuleSet,
)
from app.tools.captable import pool_status

ROOT = Path(__file__).resolve().parent.parent
LETTERS = ROOT / "data" / "docs" / "compliance"
POOL = 989_950  # unallocated pool for the seed data (spec §7)
BOARD = date(2024, 11, 15)
LETTER = "Draft Grant Letter: Test"


# --- builders ---

def rule(rid: str, field: str, op: str, value: Any, page: int, clause: str, doc: str = "ESOP Policy") -> PolicyRule:
    return PolicyRule(id=rid, field=field, op=op, value=value, doc_title=doc, page=page, clause=clause,
                      source_text=f"clause {clause} text")


RULES = [
    rule("R1", "options", "max", "$pool_remaining", 2, "3.1"),
    rule("R2", "strike_price", "min", 10, 4, "5.1"),
    rule("R3", "grant_date", "min", "$board_resolution_date", 1, "RESOLVED FURTHER THAT the Board shall administer",
         doc="Board Resolution: ESOP Pool"),
    rule("R4", "cliff_months", "equals", 12, 3, "4.2"),
    rule("R5", "total_months", "equals", 48, 3, "4.1"),
    rule("R6", "vesting_frequency", "equals", "monthly", 3, "4.3"),
    rule("R7", "exercise_window_days", "min", 90, 5, "6.1"),
    rule("R8", "acceleration_on_acquisition", "equals", 50, 7, "8.2"),
    rule("R9", "leaver_terms.unvested_lapse_on_leaving", "equals", True, 6, "7.1"),
    rule("R10", "leaver_terms.bad_leaver_forfeits_vested", "equals", True, 6, "7.4"),
]


def clean_terms(**overrides: Any) -> GrantTerms:
    """Ananya-like terms that satisfy RULES; override fields (None = letter silent)."""
    terms = {
        "options": IntTerm(value=3000, page=1, source_text="Number of Options 3,000"),
        "strike_price": NumberTerm(value=15, page=1, source_text="Exercise Price Rs 15 per Share"),
        "grant_date": DateTerm(value="2026-10-01", page=1, source_text="Grant Date 1 October 2026"),
        "cliff_months": IntTerm(value=12, page=1, source_text="Cliff 12 months"),
        "total_months": IntTerm(value=48, page=1, source_text="Total vesting period 48 months"),
        "vesting_frequency": FrequencyTerm(value="monthly", page=1, source_text="Monthly, in equal instalments"),
        "exercise_window_days": IntTerm(value=90, page=1, source_text="90 days after the Last Working Day"),
        "acceleration_on_acquisition": NumberTerm(value=50, page=1, source_text="50% of Unvested Options vest"),
        "leaver_terms": LeaverTerm(value=LeaverTerms(unvested_lapse_on_leaving=True, bad_leaver_forfeits_vested=True),
                                   page=2, source_text="you forfeit all Vested and Unvested Options"),
    }
    terms.update(overrides)
    return GrantTerms(**terms)


def by_field(findings: list[Finding]) -> dict[str, Finding]:
    return {f.field: f for f in findings}


def issues(findings: list[Finding]) -> dict[str, str]:
    return {f.field: f.status for f in findings if f.status != "match"}


# --- compare(): pure code, hand-built terms ---

def test_clean_letter_is_all_match_in_schema_order() -> None:
    findings = compare(clean_terms(), RULES, POOL, BOARD, letter_title=LETTER)
    assert [f.status for f in findings] == ["match"] * 10
    assert [f.id for f in findings] == [f"F{i}" for i in range(1, 11)]
    assert findings[0].field == "options" and findings[-1].field == "leaver_terms.bad_leaver_forfeits_vested"
    assert all(f.policy_citations and f.letter_citation.page for f in findings)


def test_vikram_like_letter_has_two_conflicts_with_both_citations() -> None:
    terms = clean_terms(cliff_months=IntTerm(value=6, page=1, source_text="Cliff 6 months"),
                        exercise_window_days=IntTerm(value=30, page=1, source_text="30 days after"))
    findings = by_field(compare(terms, RULES, POOL, BOARD, letter_title="Draft Grant Letter: Vikram Nair"))
    assert issues(list(findings.values())) == {"cliff_months": "conflict", "exercise_window_days": "conflict"}
    cliff = findings["cliff_months"]
    assert cliff.message == ("Conflict: cliff is 6 months in the letter [Draft Grant Letter: Vikram Nair, p. 1] "
                             "but the policy requires 12 months [ESOP Policy, p. 3, clause 4.2].")
    window = findings["exercise_window_days"]
    assert window.requirement == "at least 90 days" and window.rule_ids == ["R7"]
    assert window.policy_citations[0].label == "[ESOP Policy, p. 5, clause 6.1]"


def test_neha_like_letter_has_missing_window_and_exceeds_pool() -> None:
    terms = clean_terms(options=IntTerm(value=1_200_000, page=1, source_text="Number of Options 1,200,000"),
                        exercise_window_days=None)
    findings = by_field(compare(terms, RULES, POOL, BOARD, letter_title=LETTER))
    assert issues(list(findings.values())) == {"options": "exceeds_pool", "exercise_window_days": "missing"}
    pool = findings["options"]
    assert pool.requirement == "at most 989,950 options (unallocated pool, from the cap table)"
    assert pool.letter_citation.label == f"[{LETTER}, p. 1]" and pool.policy_citations[0].clause == "3.1"
    missing = findings["exercise_window_days"]
    assert missing.letter_text == "not stated" and missing.letter_citation.label == f"[{LETTER}]"
    assert missing.policy_citations[0].clause == "6.1"


@pytest.mark.parametrize(("options", "status"), [(POOL, "match"), (POOL + 1, "exceeds_pool"), (1, "match")])
def test_pool_boundary(options: int, status: str) -> None:
    terms = clean_terms(options=IntTerm(value=options, page=1, source_text="x"))
    assert by_field(compare(terms, RULES, POOL, BOARD))["options"].status == status


@pytest.mark.parametrize(("grant_date", "status"), [("2024-11-15", "match"), ("2024-11-14", "conflict")])
def test_grant_date_against_board_resolution(grant_date: str, status: str) -> None:
    terms = clean_terms(grant_date=DateTerm(value=grant_date, page=1, source_text="Grant Date"))
    finding = by_field(compare(terms, RULES, POOL, BOARD))["grant_date"]
    assert finding.status == status and finding.policy_citations[0].doc_title == "Board Resolution: ESOP Pool"
    assert finding.requirement == "on or after 15 November 2024 (board resolution date)"


def test_strike_price_minimum_and_exercise_window_above_minimum() -> None:
    terms = clean_terms(strike_price=NumberTerm(value=9.5, page=1, source_text="Rs 9.5"),
                        exercise_window_days=IntTerm(value=120, page=1, source_text="120 days"))
    assert issues(compare(terms, RULES, POOL, BOARD)) == {"strike_price": "conflict"}


def test_no_rule_means_not_covered_or_nothing() -> None:
    rules = [r for r in RULES if r.field not in ("acceleration_on_acquisition", "total_months")]
    findings = by_field(compare(clean_terms(total_months=None), rules, POOL, BOARD))
    assert "total_months" not in findings  # no rule and the letter is silent: nothing to report
    accel = findings["acceleration_on_acquisition"]
    assert accel.status == "not_covered" and accel.policy_citations == [] and accel.requirement is None


def test_silent_leaver_clause_is_missing_twice() -> None:
    findings = compare(clean_terms(leaver_terms=None), RULES, POOL, BOARD)
    assert issues(findings) == {"leaver_terms.unvested_lapse_on_leaving": "missing",
                                "leaver_terms.bad_leaver_forfeits_vested": "missing"}


def test_leaver_fact_that_contradicts_policy_is_a_conflict() -> None:
    leaver = LeaverTerm(value=LeaverTerms(unvested_lapse_on_leaving=True, bad_leaver_forfeits_vested=False),
                        page=2, source_text="a Bad Leaver keeps Vested Options")
    findings = compare(clean_terms(leaver_terms=leaver), RULES, POOL, BOARD)
    assert issues(findings) == {"leaver_terms.bad_leaver_forfeits_vested": "conflict"}


def test_several_rules_on_one_field_give_one_finding_citing_the_failed_ones() -> None:
    rules = [*RULES, rule("R11", "cliff_months", "min", 12, 3, "4.4")]
    ok = by_field(compare(clean_terms(), rules, POOL, BOARD))["cliff_months"]
    assert ok.status == "match" and ok.rule_ids == ["R4", "R11"]
    terms = clean_terms(cliff_months=IntTerm(value=6, page=1, source_text="Cliff 6 months"))
    bad = by_field(compare(terms, rules, POOL, BOARD))["cliff_months"]
    assert bad.status == "conflict" and bad.rule_ids == ["R4", "R11"]
    assert bad.requirement == "12 months and at least 12 months"


def test_compare_is_deterministic_and_counts_every_status() -> None:
    terms = clean_terms(exercise_window_days=None)
    assert compare(terms, RULES, POOL, BOARD) == compare(terms, RULES, POOL, BOARD)
    assert count_statuses(compare(terms, RULES, POOL, BOARD)) == {
        "match": 9, "conflict": 0, "missing": 1, "not_covered": 0, "exceeds_pool": 0}


def test_empty_letter_is_all_missing() -> None:
    findings = compare(GrantTerms(), RULES, POOL, BOARD)
    assert {f.status for f in findings} == {"missing"} and len(findings) == 10


# --- schema and the review gate ---

def test_date_term_must_be_iso() -> None:
    with pytest.raises(ValidationError):
        DateTerm(value="1 October 2026")


@pytest.mark.parametrize(("field", "op", "value"), [
    ("cliff_months", "equals", "12"),                     # string for an int field
    ("cliff_months", "equals", True),                     # bool is not an int here
    ("vesting_frequency", "min", "monthly"),              # min on a text field
    ("vesting_frequency", "equals", "weekly"),            # not an allowed frequency
    ("options", "min", "$pool_remaining"),                # reference with the wrong operator
    ("cliff_months", "max", "$board_resolution_date"),    # reference on the wrong field
    ("grant_date", "min", "15 Nov 2024"),                 # not ISO
])
def test_invalid_rules_are_rejected(field: str, op: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        rule("R1", field, op, value, 1, "1.1")


def rule_set(status: str = "reviewed", rules: list[PolicyRule] = RULES, reviewer: str | None = "Kautilya") -> RuleSet:
    return RuleSet(status=status, reviewed_by=reviewer, generated_by="test", source_docs=[], rules=rules)


def test_rule_set_review_gate() -> None:
    require_loadable(rule_set())
    with pytest.raises(ValueError, match="not reviewed"):
        require_loadable(rule_set("proposed", reviewer=None))
    with pytest.raises(ValueError, match=r"\$pool_remaining"):
        require_loadable(rule_set(rules=[r for r in RULES if r.id != "R1"]))
    with pytest.raises(ValidationError, match="reviewed_by"):
        rule_set(reviewer=None)
    with pytest.raises(ValidationError, match="unique"):
        rule_set(rules=[RULES[0], RULES[0]])


# --- extraction and grounding (fake LLM) ---

class FakeStructured:
    def __init__(self, owner: "FakeLLM", schema: type) -> None:
        self.owner, self.schema = owner, schema

    def invoke(self, messages: list[Any]) -> Any:
        self.owner.calls.append(("structured", self.schema.__name__, messages))
        if self.owner.error:
            raise self.owner.error
        return self.owner.terms


class FakeReply:
    def __init__(self, content: Any) -> None:
        self.content = content


class FakeLLM:
    """with_structured_output(...).invoke -> `terms`; invoke -> `report_text`. Counts calls."""

    model = "fake-model"

    def __init__(self, terms: Any = None, report_text: str = "", error: Exception | None = None,
                 report_error: Exception | None = None) -> None:
        self.terms, self.report_text, self.error, self.report_error = terms, report_text, error, report_error
        self.calls: list[Any] = []

    def with_structured_output(self, schema: type) -> FakeStructured:
        return FakeStructured(self, schema)

    def invoke(self, messages: list[Any]) -> FakeReply:
        self.calls.append(("text", messages))
        if self.report_error:
            raise self.report_error
        return FakeReply(self.report_text)


def test_extract_terms_sends_pages_and_returns_the_model() -> None:
    llm = FakeLLM(terms=clean_terms().model_dump())  # a provider returning a dict is validated into GrantTerms
    terms = extract_terms(LETTERS / "grant_letter_vikram.pdf", llm)  # type: ignore[arg-type]
    assert isinstance(terms, GrantTerms) and terms.options.value == 3000
    [(_, schema, messages)] = llm.calls
    assert schema == "GrantTerms" and "never infer" in messages[0].content.lower()
    assert "=== Page 1 ===" in messages[1].content and "=== Page 2 ===" in messages[1].content
    assert "Cliff\n6 months" in messages[1].content


def test_grounding_corrects_pages_and_drops_unquoted_values() -> None:
    pages = [(1, "Cliff\n6 months\nNumber of Options\n2,000"), (2, "within 30 days after your Last Working Day")]
    terms = clean_terms(
        cliff_months=IntTerm(value=6, page=2, source_text="Cliff 6 months"),          # wrong page
        options=IntTerm(value=2000, page=1, source_text=None),                         # no quote
        exercise_window_days=IntTerm(value=30, page=2, source_text="within 30 days"),  # fine
        total_months=IntTerm(value=48, page=1, source_text="48 months total"),         # not in the letter
    )
    grounded = ground_terms(terms, pages)
    assert grounded.terms.cliff_months.page == 1 and grounded.terms.options is None
    assert grounded.terms.exercise_window_days.page == 2 and grounded.terms.total_months.value == 48
    assert "options: dropped, the model gave a value without a quote from the letter" in grounded.warnings
    assert "cliff_months: page corrected from 2 to 1 (where the quote is)" in grounded.warnings
    assert "total_months: quote not found in the letter, check this value by hand" in grounded.warnings
    assert not any(w.startswith("exercise_window_days") for w in grounded.warnings)
    assert terms.cliff_months.page == 2  # the input is not modified


def test_letter_prompt_marks_pages() -> None:
    assert letter_prompt([(1, "a"), (2, "b")]) == "=== Page 1 ===\na\n\n=== Page 2 ===\nb"


# --- report: rephrase only, enforced in code ---

def neha_findings() -> list[Finding]:
    terms = clean_terms(options=IntTerm(value=1_200_000, page=1, source_text="1,200,000"), exercise_window_days=None)
    return compare(terms, RULES, POOL, BOARD, letter_title="Draft Grant Letter: Neha Gupta")


def good_text(findings: list[Finding]) -> str:
    lines = {"F1": "The letter grants 1,200,000 options, more than the 989,950 left in the pool.",
             "F7": "The letter does not say how long Neha has to exercise after leaving; it must be at least 90 days."}
    return "\n".join(f"[{f.id}] {lines.get(f.id, 'This term is in line with the policy.')}" for f in findings)


def test_valid_llm_text_becomes_the_report_with_code_added_labels_and_citations() -> None:
    findings = neha_findings()
    report = build_report(findings, good_text(findings))
    assert report.source == "llm" and report.problems == [] and len(report.lines) == len(findings)
    first = report.lines[0]
    assert first.finding_id == "F1" and first.text.startswith("Exceeds pool: The letter grants 1,200,000")
    assert first.text.endswith("[Draft Grant Letter: Neha Gupta, p. 1] [ESOP Policy, p. 2, clause 3.1]")


@pytest.mark.parametrize(("mutate", "problem"), [
    (lambda t: t + "\n[F11] The company should also review taxes.", "unknown finding F11"),
    (lambda t: t + "\nOverall the letter needs work.", "line without a finding id"),
    (lambda t: "\n".join(t.splitlines()[1:]), "missing from the report: F1"),
    (lambda t: t + "\n" + t.splitlines()[0], "F1 appears more than once"),
    (lambda t: t.replace("989,950 left", "900,000 left"), "numbers not in the finding: ['900000']"),
])
def test_invalid_llm_text_falls_back_to_the_template(mutate: Any, problem: str) -> None:
    findings = neha_findings()
    report = build_report(findings, mutate(good_text(findings)))
    assert report.source == "template" and any(problem in p for p in report.problems)
    assert [line.text for line in report.lines] == [f.message for f in findings]


def test_pool_line_gets_the_cap_table_note_from_code() -> None:
    findings = neha_findings()
    first = build_report(findings, good_text(findings)).lines[0]
    assert "Limit: 989,950 options (unallocated pool, from the cap table). [Draft Grant Letter" in first.text
    assert all(POOL_NOTE not in line.text for line in build_report(findings, good_text(findings)).lines[1:])


@pytest.mark.parametrize("sentence", [
    "The letter grants 1,200,000 options, exceeding the policy limit of 989,950 options.",  # real Groq report-v1 text
    "The letter grants 1,200,000 options, more than the 989,950 the Policy allows.",
    "The grant exceeds the policy limit.",  # no number, still misattributed
])
def test_report_that_calls_the_pool_a_policy_limit_is_rejected(sentence: str) -> None:
    findings = neha_findings()
    text = good_text(findings).replace(good_text(findings).splitlines()[0], f"[F1] {sentence}")
    report = build_report(findings, text)
    assert report.source == "template"
    assert any("attributes the unallocated pool (989,950 options) to the policy" in p for p in report.problems)


def test_pool_findings_are_never_worded_as_policy_limits() -> None:
    for options in (3000, 1_200_000):  # match and exceeds_pool
        terms = clean_terms(options=IntTerm(value=options, page=1, source_text=f"{options:,}"))
        finding = by_field(compare(terms, RULES, POOL, BOARD, letter_title=LETTER))["options"]
        assert finding.pool_limit == POOL
        assert "cap table limit: at most 989,950 options (unallocated pool, from the cap table)" in facts_line(finding)
        wording = finding.message
        for cite in [finding.letter_citation, *finding.policy_citations]:
            wording = wording.replace(cite.label, "")
        assert "policy" not in wording.lower() and POOL_NOTE in wording
    other = by_field(compare(clean_terms(), RULES, POOL, BOARD))["cliff_months"]
    assert other.pool_limit is None and "policy requires: 12 months" in facts_line(other)


def test_no_llm_text_is_the_template() -> None:
    report = template_report(neha_findings())
    assert report.source == "template" and report.lines[0].text.startswith("Exceeds pool: the letter grants")


def test_facts_line_and_validation_accept_dates_written_out() -> None:
    finding = by_field(neha_findings())["grant_date"]
    assert facts_line(finding).startswith(f"[{finding.id}] status: match | term: grant date | letter: 1 October 2026")
    _, problems = validate_report(f"[{finding.id}] The grant date, 1 October 2026, is after 15 November 2024.",
                                  [finding])
    assert problems == []


# --- pipeline and cache (fake LLM factory) ---

@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


class Factory:
    """llm_factory that counts how often a model was built."""

    def __init__(self, llm: FakeLLM) -> None:
        self.llm, self.built = llm, 0

    def __call__(self) -> FakeLLM:
        self.built += 1
        return self.llm


def vikram_terms() -> GrantTerms:
    return clean_terms(cliff_months=IntTerm(value=6, page=1, source_text="Cliff 6 months"),
                       exercise_window_days=IntTerm(value=30, page=1, source_text="30 days after the Last Working Day"))


def run_check(factory: Any, cache_dir: Path, rules: list[PolicyRule] = RULES, refresh: bool = False) -> Any:
    from app.compliance.pipeline import check_letter

    return check_letter(LETTERS / "grant_letter_vikram.pdf", rules, POOL, BOARD, llm_factory=factory,
                        refresh=refresh, cache_dir=cache_dir)


def test_first_run_calls_twice_and_caches_rerun_calls_never(tmp_cache: Path) -> None:
    from app.compliance.pipeline import planned_calls

    path = LETTERS / "grant_letter_vikram.pdf"
    assert planned_calls(path, RULES, POOL, BOARD, cache_dir=tmp_cache) == 2
    factory = Factory(FakeLLM(terms=vikram_terms(), report_text="not a valid report"))
    first = run_check(factory, tmp_cache)
    assert first.llm_calls == 2 and first.cached == {"extraction": False, "report": False}
    assert first.outcome == "issues_found" and first.counts["conflict"] == 2
    assert first.report.source == "template"  # the fake's text fails validation
    entry = json.loads((tmp_cache / f"{cache_module.file_hash(path)}.json").read_text())
    assert entry["extraction"]["terms"]["cliff_months"]["value"] == 6 and len(entry["reports"]) == 1

    def no_llm() -> Any:
        raise AssertionError("a cached run must not build a model")

    assert planned_calls(path, RULES, POOL, BOARD, cache_dir=tmp_cache) == 0
    again = run_check(no_llm, tmp_cache)
    assert again.llm_calls == 0 and again.cached == {"extraction": True, "report": True}
    assert again.findings == first.findings


def test_refresh_ignores_the_cache(tmp_cache: Path) -> None:
    factory = Factory(FakeLLM(terms=vikram_terms()))
    run_check(factory, tmp_cache)
    assert run_check(factory, tmp_cache, refresh=True).llm_calls == 2


def test_new_rules_rephrase_the_report_but_never_reextract(tmp_cache: Path) -> None:
    llm = FakeLLM(terms=vikram_terms())
    run_check(Factory(llm), tmp_cache)
    looser = [r for r in RULES if r.field != "cliff_months"]
    result = run_check(Factory(llm), tmp_cache, rules=looser)
    assert result.llm_calls == 1 and result.cached == {"extraction": True, "report": False}
    assert [c[0] for c in llm.calls] == ["structured", "text", "text"]


def test_report_call_failure_falls_back_and_flags_quota(tmp_cache: Path) -> None:
    llm = FakeLLM(terms=vikram_terms(), report_error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    result = run_check(Factory(llm), tmp_cache)
    assert result.report.source == "template" and result.quota_hit
    assert result.report.problems == ["report LLM call failed (rate limit / quota)"]
    assert result.llm_calls == 1  # only the extraction succeeded
    # The failed report is not cached, so the next run tries again (1 call), extraction stays cached.
    from app.compliance.pipeline import planned_calls

    assert planned_calls(LETTERS / "grant_letter_vikram.pdf", RULES, POOL, BOARD, cache_dir=tmp_cache) == 1


def test_extraction_failure_propagates_and_caches_nothing(tmp_cache: Path) -> None:
    llm = FakeLLM(error=RuntimeError("429 Too Many Requests"))
    with pytest.raises(RuntimeError) as info:
        run_check(Factory(llm), tmp_cache)
    assert cache_module.is_quota_error(info.value)
    assert not tmp_cache.exists() or not list(tmp_cache.iterdir())


# --- precision / recall and the rules script (pure parts) ---

def test_precision_recall() -> None:
    from scripts.check_letters import PLANTED, score

    assert score(set(PLANTED), set(PLANTED)) == {"flagged": 4, "planted": 4, "true_positives": 4,
                                                 "precision": 1.0, "recall": 1.0}
    extra = {("grant_letter_ananya.pdf", "total_months", "conflict")}
    three = set(sorted(PLANTED)[:3])
    s = score(three | extra, three)
    assert (s["precision"], s["recall"]) == (0.75, 0.75)


def test_proposed_rule_values_are_typed_and_bad_ones_rejected() -> None:
    from app.compliance.schema import ProposedRule, ProposedRules
    from scripts.build_policy_rules import parse_value, to_rule_set

    assert parse_value("strike_price", "10") == 10.0 and parse_value("cliff_months", "12") == 12
    assert parse_value("leaver_terms.unvested_lapse_on_leaving", "True") is True
    assert parse_value("options", "$pool_remaining") == "$pool_remaining"
    assert parse_value("acceleration_on_acquisition", "50%") == 50.0

    def proposed(field: str, op: str, value: str) -> ProposedRule:
        return ProposedRule(field=field, op=op, value=value, doc_title="ESOP Policy", page=3, clause="4.2",
                            source_text="...")

    out = to_rule_set(ProposedRules(rules=[proposed("cliff_months", "equals", "12"),
                                           proposed("cliff_months", "equals", "twelve"),
                                           proposed("total_months", "equals", "48")]), "fake", [])
    assert [r.id for r in out.rules] == ["R1", "R2"] and out.status == "proposed"
    assert out.rejected_proposals[0]["proposal"]["value"] == "twelve"


# --- integration: recorded real extractions + the reviewed rules (spec §8.7 test letters) ---

def reviewed_rules() -> list[PolicyRule]:
    if not RULES_PATH.exists():
        pytest.skip("data/policy_rules.json not generated yet")
    rule_set = load_rule_file()
    if rule_set.status != "reviewed":
        pytest.skip("data/policy_rules.json is awaiting human review")
    return rule_set.rules


def seed_pool(seed: dict[str, Any]) -> int:
    return pool_status(seed["grants"], seed["company"]["esop_pool_size"])["unallocated"]


EXPECTED = {
    "grant_letter_ananya.pdf": {},
    "grant_letter_vikram.pdf": {"cliff_months": "conflict", "exercise_window_days": "conflict"},
    "grant_letter_neha.pdf": {"exercise_window_days": "missing", "options": "exceeds_pool"},
}


@pytest.mark.parametrize("letter", list(EXPECTED))
def test_test_letters_give_exactly_the_spec_findings(letter: str, seed: dict[str, Any]) -> None:
    from app.compliance.pipeline import check_letter, cached_terms

    rules = reviewed_rules()
    path = LETTERS / letter
    entry = cache_module.load_entry(cache_module.file_hash(path))
    assert cached_terms(entry) is not None, f"no recorded extraction for {letter}: run scripts/check_letters.py"
    pool, board = seed_pool(seed), date.fromisoformat(seed["company"]["esop_board_resolution_date"])
    assert pool == POOL

    def no_llm() -> Any:
        raise AssertionError("integration test must replay recorded extractions")

    result = check_letter(path, rules, pool, board, llm_factory=no_llm)
    assert result.cached["extraction"] and result.warnings == []
    assert issues(result.findings) == EXPECTED[letter]
    for finding in result.findings:
        if finding.status != "match":
            assert finding.policy_citations, finding
            assert finding.letter_citation.page is not None or finding.status == "missing", finding
    assert result.outcome == ("compliant" if not EXPECTED[letter] else "issues_found")


# --- POST /compliance/check (fakes via dependency_overrides) ---

PDF = b"%PDF-1.4 draft letter"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    try:
        from tests.api_support import forbid_network, make_api
    except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError
        pytest.skip(f"config unavailable: {exc}")
    forbid_network(monkeypatch)
    yield from make_api()


def fake_result(terms: GrantTerms, title: str = "Draft Grant Letter: Vikram Nair") -> Any:
    from app.compliance.pipeline import CheckResult

    findings = compare(terms, RULES, POOL, BOARD, letter_title=title)
    return CheckResult(file_hash="ab" * 32, file_name="vikram.pdf", letter_title=title, terms=terms, warnings=[],
                       findings=findings, report=template_report(findings), llm_calls=0,
                       cached={"extraction": True, "report": False})


def post_letter(api: Any, user: str | None = "u_arjun", data: bytes = PDF, name: str = "../vikram draft.pdf") -> Any:
    return api.client.post("/compliance/check", files={"file": (name, data, "application/pdf")},
                           headers=api.as_user(user))


def test_check_returns_findings_report_and_writes_an_audit_record(api: Any) -> None:
    api.compliance.rules, api.compliance.result = RULES, fake_result(vikram_terms())
    response = post_letter(api)
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "issues_found" and body["counts"]["conflict"] == 2
    assert body["summary"] == "Draft Grant Letter: Vikram Nair: 8 match, 2 conflict"
    cliff = next(f for f in body["findings"] if f["field"] == "cliff_months")
    assert cliff["policy_citations"][0]["label"] == "[ESOP Policy, p. 3, clause 4.2]"
    assert body["report"]["source"] == "template" and len(body["report"]["lines"]) == 10
    [call] = api.compliance.calls
    assert call["bytes"] == PDF and call["filename"] == "vikram_draft.pdf"  # safe name, never a path
    assert call["pool_remaining"] == POOL and call["board_resolution_date"] == BOARD  # from the repo (seed data)
    [record] = api.audit.records
    assert response.headers["X-Audit-Id"] == record["id"]
    assert record["kind"] == "compliance_check" and record["outcome"] == "issues_found"
    assert record["user_id"] == "u_arjun" and record["question"] == "Compliance check: vikram_draft.pdf"
    assert record["compliance"]["counts"]["conflict"] == 2 and record["flags"] == ["report_template_fallback"]
    assert {f["status"] for f in record["compliance"]["findings"]} == {"match", "conflict"}


def test_clean_letter_is_compliant(api: Any) -> None:
    api.compliance.rules, api.compliance.result = RULES, fake_result(clean_terms(), "Draft Grant Letter: Ananya")
    body = post_letter(api).json()
    assert body["outcome"] == "compliant" and api.audit.records[0]["outcome"] == "compliant"


@pytest.mark.parametrize(("user", "status"), [(None, 401), ("u_priya", 403), ("u_nobody", 401)])
def test_check_is_admin_only(api: Any, user: str | None, status: int) -> None:
    api.compliance.rules = RULES
    assert post_letter(api, user).status_code == status
    assert api.compliance.calls == [] and api.audit.records == []


def test_non_pdf_is_rejected_before_any_work(api: Any) -> None:
    api.compliance.rules = RULES
    assert post_letter(api, data=b"hello").status_code == 415 and api.compliance.calls == []


def test_no_reviewed_rules_is_a_409(api: Any) -> None:
    response = post_letter(api)
    assert response.status_code == 409 and "build_policy_rules.py --load" in response.json()["detail"]


def test_checker_failure_is_a_500_and_audited_as_error(api: Any) -> None:
    from app.main import COMPLIANCE_ERROR

    api.compliance.rules, api.compliance.error = RULES, RuntimeError("429 RESOURCE_EXHAUSTED")
    response = post_letter(api)
    assert response.status_code == 500 and response.json()["detail"] == COMPLIANCE_ERROR
    [record] = api.audit.records
    assert record["outcome"] == "error" and "RESOURCE_EXHAUSTED" in record["error"]


def test_audit_failure_fails_closed(api: Any) -> None:
    api.compliance.rules, api.compliance.result = RULES, fake_result(vikram_terms())
    api.audit.fail = True
    assert post_letter(api).status_code == 500


def test_compliance_records_are_listed_by_get_audit(api: Any) -> None:
    api.compliance.rules, api.compliance.result = RULES, fake_result(vikram_terms())
    post_letter(api)
    [row] = api.client.get("/audit", headers=api.as_user("u_arjun")).json()
    assert row["kind"] == "compliance_check" and row["outcome"] == "issues_found"
    assert row["compliance"]["file_hash"] == "ab" * 32
