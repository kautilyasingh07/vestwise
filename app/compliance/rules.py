"""Policy rules: the human-reviewed file and the `policy_rules` collection (spec §8.7 step 2, FR-22).

Life cycle:
    scripts/build_policy_rules.py          LLM proposes -> data/policy_rules.json (status "proposed")
    a person edits the file                corrects rules, sets status "reviewed" and reviewed_by
    scripts/build_policy_rules.py --load   load_rule_file -> require_loadable -> store_rules (Mongo)
    POST /compliance/check                 get_rules(company_id)

Only a reviewed file can be loaded: rules that decide compliance must be checked by a person.
"""

from datetime import UTC, datetime
from pathlib import Path

from pymongo import ASCENDING

from app import db
from app.compliance.schema import BOARD_RESOLUTION_DATE, POOL_REMAINING, PolicyRule, RuleSet

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "policy_rules.json"


def load_rule_file(path: Path = RULES_PATH) -> RuleSet:
    """Parse and validate data/policy_rules.json (raises pydantic.ValidationError on a bad file)."""
    return RuleSet.model_validate_json(path.read_text(encoding="utf-8"))


def require_loadable(rule_set: RuleSet) -> None:
    """Raise ValueError unless the rule set is reviewed and has the two data checks (FR-24, §8.7 step 4)."""
    if rule_set.status != "reviewed":
        raise ValueError('policy_rules.json is not reviewed yet: check every rule, then set "status": "reviewed" '
                         'and "reviewed_by": "<your name>"')
    values = {r.value for r in rule_set.rules}
    for ref, what in ((POOL_REMAINING, "options max (grant must fit the unallocated pool)"),
                      (BOARD_RESOLUTION_DATE, "grant_date min (no grant before the board resolution)")):
        if ref not in values:
            raise ValueError(f"no rule uses {ref}: add a rule for {what}")


def store_rules(company_id: str, rule_set: RuleSet) -> int:
    """Replace the company's rules in `policy_rules` with this reviewed set; return how many were stored."""
    require_loadable(rule_set)
    collection = db.policy_rules()
    collection.create_index([("company_id", ASCENDING)])
    loaded_at = datetime.now(UTC)
    docs = [{"_id": f"{company_id}_{r.id}", "company_id": company_id, **r.model_dump(),
             "reviewed_by": rule_set.reviewed_by, "loaded_at": loaded_at} for r in rule_set.rules]
    collection.delete_many({"company_id": company_id})
    if docs:
        collection.insert_many(docs)
    return collection.count_documents({"company_id": company_id})


def get_rules(company_id: str) -> list[PolicyRule]:
    """The company's reviewed rules, in id order (R1, R2, ..., R10)."""
    fields = set(PolicyRule.model_fields)
    rows = db.policy_rules().find({"company_id": company_id})
    rules = [PolicyRule.model_validate({k: v for k, v in row.items() if k in fields}) for row in rows]
    return sorted(rules, key=lambda r: int(r.id[1:]))
