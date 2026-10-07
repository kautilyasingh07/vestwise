"""Compare a grant letter's terms with the policy rules and the cap table, in code (spec §8.7 steps 3-4, FR-23, FR-24).

No LLM here. Same terms + same rules + same pool -> same findings, every time,
which is what makes a verdict auditable.

One finding per letter field, in schema order:

    no rule for the field, letter silent  -> no finding (nothing to say)
    no rule for the field, letter states  -> not_covered
    rules exist, letter silent            -> missing
    every rule satisfied                  -> match
    a pool rule ($pool_remaining) fails   -> exceeds_pool
    any other rule fails                  -> conflict

Rule values `$pool_remaining` and `$board_resolution_date` are references: they
are filled in from the arguments at check time, so the cap table and the board
resolution are checked by the same loop as the policy.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, computed_field

from app.compliance.schema import (
    BOARD_RESOLUTION_DATE,
    FIELD_PATHS,
    FIELD_TYPES,
    POOL_REMAINING,
    GrantTerms,
    PolicyRule,
    RuleValue,
    letter_value,
)

Status = Literal["match", "conflict", "missing", "not_covered", "exceeds_pool"]
ISSUE_STATUSES: frozenset[str] = frozenset({"conflict", "missing", "not_covered", "exceeds_pool"})

# The pool number comes from the cap table, not the policy; this note says so wherever the number appears.
POOL_NOTE = "(unallocated pool, from the cap table)"

FIELD_LABELS: dict[str, str] = {
    "options": "number of options",
    "strike_price": "exercise price",
    "grant_date": "grant date",
    "cliff_months": "cliff",
    "total_months": "total vesting period",
    "vesting_frequency": "vesting frequency",
    "exercise_window_days": "exercise window after leaving",
    "acceleration_on_acquisition": "acceleration on acquisition",
    "leaver_terms.unvested_lapse_on_leaving": "unvested options lapse on leaving",
    "leaver_terms.bad_leaver_forfeits_vested": "bad leaver forfeits vested options",
}


class Citation(BaseModel):
    """One source of a finding: a page (and clause) of a document."""

    doc_title: str
    page: int | None = None
    clause: str | None = None
    source_text: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        """'[ESOP Policy, p. 3, clause 4.2]'; '[Draft Grant Letter: Neha Gupta]' when no page applies."""
        parts = [self.doc_title]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.clause:
            parts.append(f"clause {self.clause}")
        return "[" + ", ".join(parts) + "]"


class Finding(BaseModel):
    """The verdict on one letter field, with both sources."""

    id: str
    field: str
    label: str
    status: Status
    letter_value: bool | int | float | str | None  # dates as ISO strings
    letter_text: str  # the value as a person writes it, e.g. "6 months"
    requirement: str | None  # what the policy asks, e.g. "12 months"; None when not covered
    letter_citation: Citation
    policy_citations: list[Citation]
    rule_ids: list[str]
    message: str  # the plain template sentence, citations included
    pool_limit: int | None = None  # options left in the unallocated pool, when a $pool_remaining rule is cited


def compare(
    terms: GrantTerms,
    rules: list[PolicyRule],
    pool_remaining: int,
    board_resolution_date: date,
    *,
    letter_title: str = "Grant letter",
) -> list[Finding]:
    """Findings for every letter field (see module docstring), ids F1, F2, ... in schema order.

    Example: Vikram's letter (6-month cliff) with rule cliff_months equals 12 ->
    Finding(field="cliff_months", status="conflict", letter_text="6 months", requirement="12 months", ...)
    """
    refs: dict[str, Any] = {POOL_REMAINING: pool_remaining, BOARD_RESOLUTION_DATE: board_resolution_date}
    findings: list[Finding] = []
    for path in FIELD_PATHS:
        value, page, quote = letter_value(terms, path)
        field_rules = [r for r in rules if r.field == path]
        if not field_rules and value is None:
            continue
        status, cited = judge(path, value, field_rules, refs)
        letter_cite = Citation(doc_title=letter_title, page=page, source_text=quote)
        findings.append(make_finding(f"F{len(findings) + 1}", path, status, value, letter_cite, cited, refs))
    return findings


def judge(path: str, value: Any, rules: list[PolicyRule], refs: dict[str, Any]) -> tuple[Status, list[PolicyRule]]:
    """The status of one field and the rules to cite for it."""
    if not rules:
        return "not_covered", []
    if value is None:
        return "missing", rules
    failed = [r for r in rules if not satisfies(value, r.op, resolve(path, r.value, refs))]
    if not failed:
        return "match", rules
    if any(r.value == POOL_REMAINING for r in failed):
        return "exceeds_pool", failed
    return "conflict", failed


def resolve(path: str, value: RuleValue, refs: dict[str, Any]) -> Any:
    """A rule value ready to compare: references looked up, ISO dates parsed."""
    if isinstance(value, str) and value in refs:
        return refs[value]
    if FIELD_TYPES[path] is date:
        return date.fromisoformat(str(value))
    return value


def satisfies(actual: Any, op: str, required: Any) -> bool:
    """Does the letter's value meet one rule?"""
    if op == "min":
        return actual >= required
    if op == "max":
        return actual <= required
    if isinstance(actual, str):
        return actual.casefold() == str(required).casefold()
    return actual == required


# --- wording (used by the template report and as the facts the LLM rephrases) ---

def fmt(path: str, value: Any) -> str:
    """A value as a person writes it: 1,200,000 options; Rs 15; 1 October 2026; 6 months; yes."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, date):
        return f"{value.day} {value:%B} {value.year}"
    if path == "options":
        return f"{value:,} options"
    if path == "strike_price":
        return f"Rs {value:g}"
    if path in ("cliff_months", "total_months"):
        return f"{value} months"
    if path == "exercise_window_days":
        return f"{value} days"
    if path == "acceleration_on_acquisition":
        return f"{value:g}% of unvested options"
    return str(value)


def describe_rule(path: str, rule: PolicyRule, refs: dict[str, Any]) -> str:
    """What one rule requires: '12 months', 'at least 90 days', 'at most 989,950 options (unallocated pool)'."""
    text = fmt(path, resolve(path, rule.value, refs))
    if rule.value == POOL_REMAINING:
        text += f" {POOL_NOTE}"
    elif rule.value == BOARD_RESOLUTION_DATE:
        text += " (board resolution date)"
    if rule.op == "equals":
        return text
    if FIELD_TYPES[path] is date:
        return ("on or after " if rule.op == "min" else "on or before ") + text
    return ("at least " if rule.op == "min" else "at most ") + text


def make_finding(fid: str, path: str, status: Status, value: Any, letter_cite: Citation,
                 rules: list[PolicyRule], refs: dict[str, Any]) -> Finding:
    """Build a Finding and its template sentence.

    A limit taken from the cap table is never worded as a policy requirement: the policy
    (clause 3.1) says grants must fit the pool; the cap table says how much is left.
    """
    label = FIELD_LABELS[path]
    letter_text = fmt(path, value) if value is not None else "not stated"
    requirement = " and ".join(describe_rule(path, r, refs) for r in rules) or None
    pool_rules = [r for r in rules if r.value == POOL_REMAINING]
    pool_limit = refs[POOL_REMAINING] if pool_rules else None
    if pool_rules and len(pool_rules) == len(rules):
        source = "the ESOP pool allows"
    elif pool_rules:
        source = "the policy and the ESOP pool allow"
    else:
        source = "the policy requires"
    policy_cites = [Citation(doc_title=r.doc_title, page=r.page, clause=r.clause, source_text=r.source_text)
                    for r in rules]
    lc, pc = letter_cite.label, " ".join(c.label for c in policy_cites)
    message = {
        "match": f"Match: {label} is {letter_text} in the letter {lc}; {source} {requirement} {pc}.",
        "conflict": f"Conflict: {label} is {letter_text} in the letter {lc} but {source} {requirement} {pc}.",
        "missing": f"Missing: the letter does not state the {label} {lc}; {source} {requirement} {pc}.",
        "not_covered": f"Not covered: {label} is {letter_text} in the letter {lc}, but no policy rule covers it.",
        "exceeds_pool": f"Exceeds pool: the letter grants {letter_text} {lc} but {source} {requirement} {pc}.",
    }[status]
    json_value = value.isoformat() if isinstance(value, date) else value
    return Finding(id=fid, field=path, label=label, status=status, letter_value=json_value, letter_text=letter_text,
                   requirement=requirement, letter_citation=letter_cite, policy_citations=policy_cites,
                   rule_ids=[r.id for r in rules], message=message, pool_limit=pool_limit)


def count_statuses(findings: list[Finding]) -> dict[str, int]:
    """{'match': 8, 'conflict': 2, ...} with every status present (zeros included)."""
    counts = dict.fromkeys(Status.__args__, 0)  # type: ignore[attr-defined]
    for f in findings:
        counts[f.status] += 1
    return counts
