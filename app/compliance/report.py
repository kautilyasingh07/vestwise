"""Turn findings into a readable report (spec §8.7 step 5, FR-25).

The LLM may only rephrase. Code enforces that:

1. Every finding has an id (F1, F2, ...). The model must write exactly one line per
   finding, each starting with its id: "[F2] The cliff in this letter is 6 months ...".
2. `validate_report` rejects the text if any line has no id or an unknown id, if a
   finding is missing or repeated, if a line contains a number that isn't in its
   finding (a cheap guard against invented facts), or if a line about the ESOP pool
   mentions the policy (the pool number comes from the cap table, not the policy).
3. The status label, both citations and, for pool findings, the note
   "Limit: N options (unallocated pool, from the cap table)." are added by code, not by
   the model, so they can't be dropped or misattributed.

If the text fails any check (or the call fails), the report is the plain template
built from each finding's `message`; `problems` says why.
"""

import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.compliance.check import POOL_NOTE, Finding

PROMPT_VERSION = "report-v2"  # v2: pool limits are facts from the cap table, never "policy requires"

STATUS_LABELS = {"match": "Match", "conflict": "Conflict", "missing": "Missing",
                 "not_covered": "Not covered", "exceeds_pool": "Exceeds pool"}

SYSTEM_PROMPT = """You turn the findings of a grant letter compliance check into a short report for an HR administrator.

Each finding has an id such as F1. Write exactly one line per finding, in the same order, and start
every line with that id in square brackets, for example: [F1] The letter sets the cliff at ...

Rules:
- Only rephrase the finding in plain English. Do not add, merge, split or drop findings.
- Do not add facts, numbers, dates, advice or opinions that are not in the finding.
- A "cap table limit" is how many options are left in the unallocated ESOP pool. It is not a policy rule:
  in that line, do not mention the policy at all.
- Do not start a line with the status word (Match, Conflict, ...) and do not write citations;
  the system adds both.
- One sentence per line. Plain text: no headings, bullets, blank lines, or text before or after the lines."""

LINE_RE = re.compile(r"^\[(F\d+)\]\s*(.+)$")
POLICY_RE = re.compile(r"\bpolic(?:y|ies)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


class ReportLine(BaseModel):
    """One line of the report: always about exactly one finding."""

    finding_id: str
    status: str
    text: str  # final text: status label + sentence + citations


class Report(BaseModel):
    """The report and how it was made."""

    source: Literal["llm", "template"]
    lines: list[ReportLine]
    problems: list[str] = Field(default_factory=list)  # why the LLM text was rejected, if it was


def facts_line(finding: Finding) -> str:
    """The finding as the model sees it: '[F2] status: conflict | term: cliff | letter: 6 months | policy requires: 12 months'.

    A pool finding says 'cap table limit:' instead of 'policy requires:', so the model is never
    told that the pool number is a policy figure.
    """
    basis = "cap table limit" if finding.pool_limit is not None else "policy requires"
    requirement = finding.requirement or "no rule in the policy"
    return (f"[{finding.id}] status: {finding.status.replace('_', ' ')} | term: {finding.label} | "
            f"letter: {finding.letter_text} | {basis}: {requirement}")


def pool_note(finding: Finding) -> str:
    """' Limit: 989,950 options (unallocated pool, from the cap table).' for pool findings, else ''."""
    return "" if finding.pool_limit is None else f" Limit: {finding.pool_limit:,} options {POOL_NOTE}."


def citations_text(finding: Finding) -> str:
    """'[letter cite] [policy cite] ...' as appended to every line."""
    return " ".join([finding.letter_citation.label, *(c.label for c in finding.policy_citations)])


def numbers(text: str) -> set[str]:
    """Numbers in a text, normalised: '1,200,000' -> '1200000', '15.0' -> '15', '01' -> '1'."""
    out = set()
    for raw in NUMBER_RE.findall(text):
        value = float(raw.replace(",", ""))
        out.add(str(int(value)) if value.is_integer() else str(value))
    return out


def allowed_numbers(finding: Finding) -> set[str]:
    """Every number the model may use for a finding: those in its facts, dates' parts, pages and clauses."""
    allowed = numbers(facts_line(finding).split("]", 1)[1]) | numbers(citations_text(finding))
    if finding.field == "grant_date" and isinstance(finding.letter_value, str):
        allowed |= numbers(finding.letter_value.replace("-", " "))
    return allowed


def validate_report(text: str, findings: list[Finding]) -> tuple[dict[str, str], list[str]]:
    """Check the model's text line by line. Returns ({finding_id: sentence}, problems); no problems = usable."""
    by_id = {f.id: f for f in findings}
    phrased: dict[str, str] = {}
    problems: list[str] = []
    for line in (raw.strip() for raw in text.strip().splitlines()):
        if not line:
            continue
        match = LINE_RE.match(line)
        if match is None:
            problems.append(f"line without a finding id: {line[:80]!r}")
            continue
        fid, sentence = match.group(1), match.group(2).strip()
        if fid not in by_id:
            problems.append(f"line refers to unknown finding {fid}")
        elif fid in phrased:
            problems.append(f"finding {fid} appears more than once")
        else:
            extra = numbers(sentence) - allowed_numbers(by_id[fid])
            if extra:
                problems.append(f"{fid} mentions numbers not in the finding: {sorted(extra)}")
            pool = by_id[fid].pool_limit
            if pool is not None and POLICY_RE.search(sentence):
                problems.append(f"{fid} attributes the unallocated pool ({pool:,} options) to the policy; "
                                "it comes from the cap table")
            phrased[fid] = sentence
    missing = [f.id for f in findings if f.id not in phrased]
    if missing:
        problems.append(f"findings missing from the report: {', '.join(missing)}")
    return phrased, problems


def template_report(findings: list[Finding], problems: list[str] | None = None) -> Report:
    """The fallback: each finding's own template sentence, no LLM involved."""
    lines = [ReportLine(finding_id=f.id, status=f.status, text=f.message) for f in findings]
    return Report(source="template", lines=lines, problems=problems or [])


def build_report(findings: list[Finding], llm_text: str | None, problems: list[str] | None = None) -> Report:
    """The LLM report if `llm_text` passes validation, otherwise the template (with the reasons)."""
    if llm_text is None:
        return template_report(findings, problems)
    phrased, found = validate_report(llm_text, findings)
    if found:
        return template_report(findings, found)
    lines = [ReportLine(finding_id=f.id, status=f.status,
                        text=f"{STATUS_LABELS[f.status]}: {phrased[f.id]}{pool_note(f)} {citations_text(f)}")
             for f in findings]
    return Report(source="llm", lines=lines)


def generate_report_text(findings: list[Finding], llm: BaseChatModel) -> str:
    """One live LLM call: the findings' facts in, raw report text out (validated by build_report)."""
    facts = "\n".join(facts_line(f) for f in findings)
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Findings:\n{facts}")])
    content = reply.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)
