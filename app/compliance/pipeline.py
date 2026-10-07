"""The whole compliance check for one letter: cached extraction -> grounding -> compare -> cached report.

    extract (LLM, cached) -> ground_terms (code) -> compare (code) -> report (LLM, cached, validated)

At most two live LLM calls per letter, and zero on a rerun: both outputs are cached
under the PDF's hash (app/compliance/cache.py). The report is cached per set of
findings, so changing the rules or the pool re-phrases the report but never
re-extracts the letter. `refresh=True` ignores the cache and calls the LLM again.

The model is created through `llm_factory` only when a live call is needed, so a
fully cached run never even builds a client (tests rely on this).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.compliance import extract, report
from app.compliance.cache import CACHE_DIR, file_hash, is_quota_error, json_hash, load_entry, now_iso, save_entry
from app.compliance.check import Finding, compare, count_statuses
from app.compliance.extract import ground_terms
from app.compliance.report import Report
from app.compliance.schema import GrantTerms, PolicyRule
from app.ingest.loader import load_pdf, pdf_title

LLMFactory = Callable[[], BaseChatModel]


def default_llm() -> BaseChatModel:
    """The configured chat model with client retries off: a 429 must surface, not burn more quota."""
    from app.llm import get_chat_model

    return get_chat_model(max_retries=0)


def model_label(llm: BaseChatModel) -> str:
    """Model name for cache records (ChatGoogleGenerativeAI has `model`, ChatGroq `model_name`)."""
    return str(getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__)


@dataclass
class CheckResult:
    """Everything one check produced, plus what it cost."""

    file_hash: str
    file_name: str
    letter_title: str
    terms: GrantTerms
    warnings: list[str]
    findings: list[Finding]
    report: Report
    llm_calls: int = 0
    cached: dict[str, bool] = field(default_factory=dict)  # {"extraction": True, "report": False}
    quota_hit: bool = False  # the report call hit a 429; callers running batches should stop

    @property
    def counts(self) -> dict[str, int]:
        """Findings per status."""
        return count_statuses(self.findings)

    @property
    def outcome(self) -> Literal["compliant", "issues_found"]:
        """compliant only when every finding is a match."""
        return "compliant" if all(f.status == "match" for f in self.findings) else "issues_found"

    def summary(self) -> str:
        """One line for logs and the audit record: 'Draft Grant Letter: Vikram Nair: 8 match, 2 conflict'."""
        parts = [f"{n} {status.replace('_', ' ')}" for status, n in self.counts.items() if n]
        return f"{self.letter_title}: {', '.join(parts) or 'no findings'}"


def report_key(findings: list[Finding]) -> str:
    """Cache key of a report: the findings it phrases plus the prompt version."""
    return json_hash({"prompt": report.PROMPT_VERSION, "findings": [f.model_dump(mode="json") for f in findings]})


def cached_terms(entry: dict[str, Any]) -> GrantTerms | None:
    """The cached extraction, if there is one made with the current prompt."""
    cached = entry.get("extraction")
    if not cached or cached.get("prompt_version") != extract.PROMPT_VERSION:
        return None
    return GrantTerms.model_validate(cached["terms"])


def check_letter(
    pdf_path: Path,
    rules: list[PolicyRule],
    pool_remaining: int,
    board_resolution_date: date,
    *,
    llm_factory: LLMFactory = default_llm,
    refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> CheckResult:
    """Run the full check on one letter PDF. Extraction errors propagate; report errors fall back to the template."""
    sha = file_hash(pdf_path)
    entry = load_entry(sha, cache_dir)
    entry.update(file_hash=sha, file_name=pdf_path.name)
    llm: BaseChatModel | None = None
    calls = 0

    terms = None if refresh else cached_terms(entry)
    extraction_cached = terms is not None
    if terms is None:
        llm = llm_factory()
        terms = extract.extract_terms(pdf_path, llm)
        calls += 1
        entry["extraction"] = {"model": model_label(llm), "prompt_version": extract.PROMPT_VERSION,
                               "created_at": now_iso(), "terms": terms.model_dump(mode="json")}
        save_entry(sha, entry, cache_dir)

    grounded = ground_terms(terms, load_pdf(pdf_path))
    title = pdf_title(pdf_path)
    findings = compare(grounded.terms, rules, pool_remaining, board_resolution_date, letter_title=title)

    key = report_key(findings)
    reports = entry.setdefault("reports", {})
    report_cached = key in reports and not refresh
    quota_hit = False
    if report_cached:
        result_report = report.build_report(findings, reports[key]["text"])
    else:
        try:
            llm = llm or llm_factory()
            text = report.generate_report_text(findings, llm)
            calls += 1
            reports[key] = {"model": model_label(llm), "prompt_version": report.PROMPT_VERSION,
                            "created_at": now_iso(), "text": text}
            save_entry(sha, entry, cache_dir)
            result_report = report.build_report(findings, text)
        except Exception as exc:  # noqa: BLE001 - the findings stand; only the wording falls back
            quota_hit = is_quota_error(exc)
            reason = "rate limit / quota" if quota_hit else type(exc).__name__
            result_report = report.template_report(findings, [f"report LLM call failed ({reason})"])

    return CheckResult(file_hash=sha, file_name=pdf_path.name, letter_title=title, terms=grounded.terms,
                       warnings=grounded.warnings, findings=findings, report=result_report, llm_calls=calls,
                       cached={"extraction": extraction_cached, "report": report_cached}, quota_hit=quota_hit)


def planned_calls(
    pdf_path: Path,
    rules: list[PolicyRule],
    pool_remaining: int,
    board_resolution_date: date,
    *,
    refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> int:
    """How many live LLM calls check_letter would make right now (0, 1 or 2). Makes none itself."""
    if refresh:
        return 2
    terms = cached_terms(load_entry(file_hash(pdf_path), cache_dir))
    if terms is None:
        return 2
    grounded = ground_terms(terms, load_pdf(pdf_path))
    findings = compare(grounded.terms, rules, pool_remaining, board_resolution_date, letter_title=pdf_title(pdf_path))
    reports = load_entry(file_hash(pdf_path), cache_dir).get("reports", {})
    return 0 if report_key(findings) in reports else 1
