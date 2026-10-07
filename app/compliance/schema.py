"""Grant letter terms and policy rules: the fixed schemas of the compliance checker (spec §8.7, FR-21, FR-22).

Two families of models:

- `GrantTerms`: what one draft grant letter says. Every field from spec §8.7 is a
  `{value, page, source_text}` object and every field is optional: a letter that is
  silent on a term gets `None`, never a guess. The LLM fills this model through
  structured output (app/compliance/extract.py).
- `PolicyRule` / `RuleSet`: what the ESOP policy allows, one rule per (field, operator),
  each with the document, page and clause it comes from. Proposed once by the LLM
  (scripts/build_policy_rules.py), corrected and approved by a human, then loaded into
  the `policy_rules` collection (app/compliance/rules.py).

Rules address letter fields by *path*. `leaver_terms` is a small object, so its two
checkable facts are separate paths (`leaver_terms.unvested_lapse_on_leaving`, ...)
and each gets its own rule and citation.
"""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Frequency = Literal["monthly", "quarterly", "annually"]

# Every path a rule can check, in report order.
FieldPath = Literal[
    "options",
    "strike_price",
    "grant_date",
    "cliff_months",
    "total_months",
    "vesting_frequency",
    "exercise_window_days",
    "acceleration_on_acquisition",
    "leaver_terms.unvested_lapse_on_leaving",
    "leaver_terms.bad_leaver_forfeits_vested",
]
FIELD_PATHS: tuple[str, ...] = FieldPath.__args__  # type: ignore[attr-defined]

# Python type of each path's value: drives rule validation and comparison.
FIELD_TYPES: dict[str, type] = {
    "options": int,
    "strike_price": float,
    "grant_date": date,
    "cliff_months": int,
    "total_months": int,
    "vesting_frequency": str,
    "exercise_window_days": int,
    "acceleration_on_acquisition": float,
    "leaver_terms.unvested_lapse_on_leaving": bool,
    "leaver_terms.bad_leaver_forfeits_vested": bool,
}

# Rule values that are filled in from data at check time, not written into the rule (spec §8.7 step 4).
POOL_REMAINING = "$pool_remaining"
BOARD_RESOLUTION_DATE = "$board_resolution_date"
REFERENCES: dict[str, tuple[str, str]] = {  # reference -> the only (path, op) it may be used with
    POOL_REMAINING: ("options", "max"),
    BOARD_RESOLUTION_DATE: ("grant_date", "min"),
}

Op = Literal["equals", "min", "max"]
RuleValue = bool | int | float | str


# --- letter terms (LLM output) ---

class Term(BaseModel):
    """Where a value was found: the letter page and a short verbatim quote."""

    page: int | None = Field(default=None, description="1-based page number of the letter where this term is stated.")
    source_text: str | None = Field(
        default=None,
        description="Short verbatim quote from the letter that states this term (at most 200 characters). "
                    "Copy it exactly; do not paraphrase.")


class IntTerm(Term):
    """A whole-number term (options, months, days)."""

    value: int | None = Field(default=None, description="The number exactly as stated; null if the letter does not state it.")


class NumberTerm(Term):
    """A decimal term (price in rupees, a percentage)."""

    value: float | None = Field(default=None, description="The number exactly as stated; null if the letter does not state it.")


class DateTerm(Term):
    """A date term, as an ISO string (Gemini's JSON schema support for `format: date` is not relied on)."""

    value: str | None = Field(default=None, description="ISO date YYYY-MM-DD; null if the letter does not state it.")

    @field_validator("value")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        """Reject anything that isn't YYYY-MM-DD, so compare() can rely on it."""
        if value is not None:
            date.fromisoformat(value)
        return value


class FrequencyTerm(Term):
    """How often options vest after the cliff."""

    value: Frequency | None = Field(default=None, description="Vesting frequency after the cliff; null if not stated.")


class LeaverTerms(BaseModel):
    """The two checkable facts in a letter's leaver clause."""

    unvested_lapse_on_leaving: bool | None = Field(
        default=None, description="True if the letter says unvested options lapse when the employee leaves; "
                                  "false if it says they do not; null if it does not say.")
    bad_leaver_forfeits_vested: bool | None = Field(
        default=None, description="True if the letter says a bad leaver forfeits vested options too; "
                                  "false if it says a bad leaver keeps them; null if it does not say.")


class LeaverTerm(Term):
    """The leaver clause, reduced to facts code can compare."""

    value: LeaverTerms | None = Field(default=None, description="Null if the letter has no leaver terms.")


class GrantTerms(BaseModel):
    """Every term of a draft grant letter that the checker compares (spec §8.7 step 1). All optional."""

    options: IntTerm | None = Field(default=None, description="Number of options granted.")
    strike_price: NumberTerm | None = Field(default=None, description="Exercise (strike) price per share, in rupees.")
    grant_date: DateTerm | None = Field(default=None, description="Grant date of the options (not the letter date).")
    cliff_months: IntTerm | None = Field(default=None, description="Cliff length in months.")
    total_months: IntTerm | None = Field(default=None, description="Total vesting period in months.")
    vesting_frequency: FrequencyTerm | None = Field(default=None, description="Vesting frequency after the cliff.")
    exercise_window_days: IntTerm | None = Field(
        default=None, description="Days a (good) leaver has to exercise vested options after the last working day.")
    acceleration_on_acquisition: NumberTerm | None = Field(
        default=None, description="Percentage of unvested options that vest on a change of control / acquisition.")
    leaver_terms: LeaverTerm | None = Field(default=None, description="What happens to options when the employee leaves.")


def letter_value(terms: GrantTerms, path: str) -> tuple[Any, int | None, str | None]:
    """(value, page, source_text) of one path; value is None when the letter is silent.

    Converts to the path's Python type (ISO strings -> date), so callers compare like with like.
    """
    head, _, sub = path.partition(".")
    term = getattr(terms, head)
    if term is None or term.value is None:
        return None, None, None
    value = getattr(term.value, sub) if sub else term.value
    if value is not None and FIELD_TYPES[path] is date:
        value = date.fromisoformat(value)
    return value, term.page, term.source_text


# --- policy rules ---

class RuleBase(BaseModel):
    """Fields shared by a proposed (LLM) rule and a stored (reviewed) rule."""

    field: FieldPath = Field(description="The grant letter field this rule constrains.")
    op: Op = Field(description="equals: must be exactly the value; min: at least; max: at most.")
    doc_title: str = Field(description="Title of the document the rule comes from, e.g. 'ESOP Policy'.")
    page: int = Field(ge=1, description="1-based page of that document where the clause is.")
    clause: str = Field(description="Clause number, e.g. '4.2'.")
    source_text: str = Field(description="Short verbatim quote of the clause text that states the rule.")


class ProposedRule(RuleBase):
    """One rule as the LLM proposes it. `value` is text; build_policy_rules.py types it per field."""

    value: str = Field(description="The required value as text: a number ('12', '10.0'), an ISO date, 'monthly', "
                                   "'true'/'false', or one of the references '$pool_remaining', '$board_resolution_date'.")


class ProposedRules(BaseModel):
    """The LLM's whole proposal."""

    rules: list[ProposedRule]


class PolicyRule(RuleBase):
    """A reviewed rule, as stored in data/policy_rules.json and the policy_rules collection."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^R\d+$")
    value: RuleValue

    @model_validator(mode="after")
    def _value_fits_field(self) -> "PolicyRule":
        """The value's type must match the field; references only where they make sense."""
        check_rule_value(self.field, self.op, self.value)
        return self


def check_rule_value(path: str, op: str, value: RuleValue) -> None:
    """Raise ValueError unless `value` is a valid `op` value for `path`."""
    if isinstance(value, str) and value.startswith("$"):
        if REFERENCES.get(value) != (path, op):
            raise ValueError(f"{value} can only be used as {REFERENCES.get(value, ('?', '?'))}, not ({path}, {op})")
        return
    kind = FIELD_TYPES[path]
    if kind in (bool, str) and op != "equals":
        raise ValueError(f"{path} only supports 'equals', not {op!r}")
    if kind is bool and not isinstance(value, bool):
        raise ValueError(f"{path} needs true/false, got {value!r}")
    if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{path} needs a whole number, got {value!r}")
    if kind is float and (isinstance(value, bool) or not isinstance(value, int | float)):
        raise ValueError(f"{path} needs a number, got {value!r}")
    if kind is date:
        if not isinstance(value, str):
            raise ValueError(f"{path} needs an ISO date string, got {value!r}")
        date.fromisoformat(value)
    if kind is str and value not in Frequency.__args__:  # type: ignore[attr-defined]
        raise ValueError(f"{path} must be one of {Frequency.__args__}, got {value!r}")  # type: ignore[attr-defined]


class RuleSet(BaseModel):
    """data/policy_rules.json: the rules plus their review status (human-in-the-loop gate, FR-22)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["proposed", "reviewed"]
    reviewed_by: str | None = None
    generated_by: str
    source_docs: list[dict[str, str]]
    rules: list[PolicyRule]
    # LLM proposals that failed validation, kept for the reviewer to fix or discard (never loaded).
    rejected_proposals: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "RuleSet":
        """Rule ids are how findings point at rules; duplicates would make citations ambiguous."""
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        if self.status == "reviewed" and not self.reviewed_by:
            raise ValueError("a reviewed rule set must say who reviewed it (reviewed_by)")
        return self
