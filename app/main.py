"""FastAPI app (spec §9): chat, vesting, cap table, documents, compliance check, audit.

Identity flows one way: X-User-Id header -> `get_ctx` -> RequestContext ->
endpoint -> tools. Nothing about identity is read from a request body; the
request models (app/schemas.py) have no identity fields and ignore unknown
ones. In production the header would be replaced by a verified JWT and only
`get_ctx` would change.

Every collaborator (user lookup, agent, audit store, repo, ingestion) is a
FastAPI dependency, so tests swap them with `app.dependency_overrides` and
make no network calls.

Run:  uvicorn app.main:app --reload     then open http://localhost:8000/docs
"""

import logging
import re
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, Response, UploadFile, status
from pypdf.errors import PdfReadError

from app.agent import run_agent
from app.audit import MAX_LIMIT, classify, read_audit, write_audit, write_compliance_audit
from app.compliance.pipeline import CheckResult, check_letter
from app.compliance.rules import get_rules
from app.compliance.schema import PolicyRule
from app.config import settings
from app.context import RequestContext, UnknownUserError, load_context
from app.ingest.loader import pdf_title
from app.ingest.pipeline import PERSONAL_DOC_TYPES, DocType, IngestResult, ingest_file
from app.schemas import (
    AuditRecord,
    CapTableResponse,
    ChatRequest,
    ChatResponse,
    ComplianceResponse,
    DilutionResponse,
    DocumentResponse,
    SimulateRequest,
    VestingResponse,
)
from app.tools import repo as mongo_repo
from app.tools.captable import cap_table, pool_status, simulate_dilution
from app.tools.vesting import compute_vesting

logger = logging.getLogger("vestwise")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ERROR_CHARS = 500
CHAT_ERROR = "Something went wrong while answering. The error has been logged."

app = FastAPI(
    title="Vestwise",
    version="0.6.0",
    description="RAG + tool-calling ESOP assistant. Send `X-User-Id` (u_priya, u_rahul, u_arjun) on every request.",
)


# --- dependencies (overridable in tests) ---

UserLoader = Callable[[str], RequestContext]
AgentFn = Callable[..., dict[str, Any]]
AuditWriter = Callable[..., str]
AuditReader = Callable[[str, int], list[dict[str, Any]]]
Ingester = Callable[..., IngestResult]
RulesLoader = Callable[[str], list[PolicyRule]]
Checker = Callable[..., CheckResult]
ComplianceAuditWriter = Callable[..., str]


def get_user_loader() -> UserLoader:
    """How a user id becomes a RequestContext (Mongo `users` lookup)."""
    return load_context


def get_agent() -> AgentFn:
    """The agent that answers /chat."""
    return run_agent


def get_audit_writer() -> AuditWriter:
    """Writes one audit record per /chat request."""
    return write_audit


def get_audit_reader() -> AuditReader:
    """Reads the latest audit records of a company."""
    return read_audit


def get_repo() -> Any:
    """Data access for vesting and cap table (module app.tools.repo; a fake in tests)."""
    return mongo_repo


def get_ingester() -> Ingester:
    """The Phase 3 ingestion pipeline."""
    return ingest_file


def get_rules_loader() -> RulesLoader:
    """Reviewed policy rules of a company (Mongo `policy_rules`)."""
    return get_rules


def get_compliance_checker() -> Checker:
    """The Phase 9 compliance pipeline (cached LLM extraction + code verdicts + validated report)."""
    return check_letter


def get_compliance_audit_writer() -> ComplianceAuditWriter:
    """Writes one audit record per compliance check."""
    return write_compliance_audit


def get_ctx(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    loader: UserLoader = Depends(get_user_loader),
) -> RequestContext:
    """Authenticate: X-User-Id header -> RequestContext. 401 if missing or unknown.

    The only place identity enters the app. A user record that exists but is
    unusable (bad role, no company) is 403: we know who it is, we won't serve them.
    """
    if not x_user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-User-Id header")
    try:
        return loader(x_user_id)
    except UnknownUserError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user") from None
    except ValueError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "User record is not valid for access") from None


def require_admin(ctx: RequestContext = Depends(get_ctx)) -> RequestContext:
    """Authorise: 403 unless the authenticated user is an admin."""
    if not ctx.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return ctx


Ctx = Annotated[RequestContext, Depends(get_ctx)]
AdminCtx = Annotated[RequestContext, Depends(require_admin)]


def elapsed_ms(start: float) -> int:
    """Milliseconds since `start` (a time.monotonic() value)."""
    return round((time.monotonic() - start) * 1000)


# --- /chat ---

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    body: ChatRequest,
    response: Response,
    ctx: Ctx,
    agent: AgentFn = Depends(get_agent),
    audit: AuditWriter = Depends(get_audit_writer),
) -> ChatResponse:
    """Answer one message as the header's user. Every call is audit-logged, including failures.

    `outcome` (answered | refused | not_found) comes from the same classifier as the audit
    record. A failure is a 500 whose audit record has outcome "error". The audit record id
    is returned in the `X-Audit-Id` response header.
    """
    start = time.monotonic()
    history = [item.model_dump() for item in body.history]
    try:
        result = agent(ctx, body.message, history, body.as_of)
    except Exception as exc:  # noqa: BLE001 - any failure: log it, audit it, return a clean 500
        latency = elapsed_ms(start)
        logger.exception("chat failed for user %s", ctx.user_id)
        error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        try:
            audit(ctx, body.message, [], [], None, latency, as_of=body.as_of, error=error)
        except Exception:  # noqa: BLE001 - the original error is what the client needs to hear about
            logger.exception("audit write failed after a chat error")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, CHAT_ERROR) from None

    latency = elapsed_ms(start)
    try:
        audit_id = audit(ctx, body.message, result["chunk_ids"], result["tool_calls"], result["answer"],
                         latency, as_of=body.as_of, flags=result.get("flags", []),
                         citation_check=result.get("citation_check"))
    except Exception:  # noqa: BLE001 - fail closed: no answer leaves without an audit record
        logger.exception("audit write failed for user %s", ctx.user_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, CHAT_ERROR) from None
    response.headers["X-Audit-Id"] = audit_id
    return ChatResponse(answer=result["answer"], outcome=classify(result["answer"], None),
                        citations=result["citations"], tool_calls=result["tool_calls"], latency_ms=latency)


# --- /vesting ---

@app.get("/vesting/{stakeholder_id}", response_model=VestingResponse, tags=["data"])
def vesting(stakeholder_id: str, ctx: Ctx, as_of: date | None = None, repo: Any = Depends(get_repo)) -> VestingResponse:
    """Vesting of one stakeholder. Employees: only their own (403 otherwise). Admins: anyone in their company."""
    if not ctx.is_admin and stakeholder_id != ctx.stakeholder_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Employees can only view their own vesting")
    names = repo.get_stakeholder_names(ctx.company_id)
    if stakeholder_id not in names:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stakeholder not found in your company")
    when = as_of or date.today()
    grants = repo.get_grants(ctx.company_id, stakeholder_id)
    return VestingResponse(
        stakeholder_id=stakeholder_id, name=names[stakeholder_id], as_of=when,
        grants=[compute_vesting(g, when, settings.exercise_window_days) for g in grants],
    )


# --- /captable ---

@app.get("/captable", response_model=CapTableResponse, tags=["admin"])
def get_cap_table(ctx: AdminCtx, repo: Any = Depends(get_repo)) -> dict[str, Any]:
    """Holders with issued % and fully diluted %, plus the unallocated pool (admin only)."""
    return cap_table(repo.get_holdings(ctx.company_id), repo.get_grants(ctx.company_id),
                     repo.get_pool_size(ctx.company_id), repo.get_stakeholder_names(ctx.company_id))


@app.post("/captable/simulate", response_model=DilutionResponse, tags=["admin"])
def simulate(body: SimulateRequest, ctx: AdminCtx, repo: Any = Depends(get_repo)) -> dict[str, Any]:
    """Before/after ownership if `new_shares` go to a new investor (admin only)."""
    return simulate_dilution(repo.get_holdings(ctx.company_id), repo.get_grants(ctx.company_id),
                             repo.get_pool_size(ctx.company_id), body.new_shares, body.investor_name,
                             repo.get_stakeholder_names(ctx.company_id))


# --- /documents ---

def safe_filename(name: str | None) -> str:
    """Base name only, safe characters only, always ending in .pdf (it becomes documents.filename)."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "").name).strip("._") or "upload"
    return base if base.lower().endswith(".pdf") else f"{base}.pdf"


def read_pdf_upload(file: UploadFile) -> bytes:
    """The uploaded bytes, or 413 (over MAX_UPLOAD_BYTES) / 415 (no PDF magic bytes)."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"PDF larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload must be a PDF")
    return data


def check_document_owner(doc_type: DocType, owner: str | None, company_id: str, repo: Any) -> None:
    """Grant letters need an owner from this company; company-wide documents must not have one."""
    if doc_type in PERSONAL_DOC_TYPES:
        if not owner:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"{doc_type} needs owner_stakeholder_id (otherwise every employee could read it)")
        if owner not in repo.get_stakeholder_names(company_id):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "owner_stakeholder_id is not in your company")
    elif owner:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"{doc_type} is company-wide; only grant letters have an owner")


@app.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, tags=["admin"])
def upload_document(
    file: UploadFile,
    doc_type: Annotated[DocType, Form()],
    ctx: AdminCtx,
    owner_stakeholder_id: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    repo: Any = Depends(get_repo),
    ingest: Ingester = Depends(get_ingester),
) -> DocumentResponse:
    """Ingest a PDF into the admin's company (admin only). Re-uploading the same file replaces its chunks.

    Grant letters must name `owner_stakeholder_id`; policy and board resolutions must not.
    """
    owner = owner_stakeholder_id or None  # an empty form field means "no owner"
    check_document_owner(doc_type, owner, ctx.company_id, repo)
    data = read_pdf_upload(file)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / safe_filename(file.filename)
        path.write_bytes(data)
        try:
            result = ingest(path, ctx.company_id, doc_type, title or pdf_title(path), owner)
        except (ValueError, PdfReadError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Could not ingest: {exc}") from None
    return DocumentResponse(doc_id=result.doc_id, chunks_created=result.chunks, title=result.title,
                            pages=result.pages, owner_stakeholder_id=result.owner_stakeholder_id,
                            replaced_chunks=result.replaced_chunks)


# --- /compliance ---

NO_RULES = ("No reviewed policy rules are loaded for your company. Review data/policy_rules.json, "
            "then run: python scripts/build_policy_rules.py --load")
COMPLIANCE_ERROR = "The compliance check failed. The error has been logged."


@app.post("/compliance/check", response_model=ComplianceResponse, tags=["admin"])
def compliance_check(
    file: UploadFile,
    response: Response,
    ctx: AdminCtx,
    repo: Any = Depends(get_repo),
    load_rules: RulesLoader = Depends(get_rules_loader),
    checker: Checker = Depends(get_compliance_checker),
    audit: ComplianceAuditWriter = Depends(get_compliance_audit_writer),
) -> ComplianceResponse:
    """Check a draft grant letter against the reviewed policy rules and the cap table (admin only).

    The LLM extracts the letter's terms and phrases the report; every verdict is computed in
    code. Each check is audit-logged (outcome compliant | issues_found | error); the record id
    is in the `X-Audit-Id` header. The letter itself is not stored or ingested.
    """
    start = time.monotonic()
    data = read_pdf_upload(file)
    file_name = safe_filename(file.filename)
    rules = load_rules(ctx.company_id)
    if not rules:
        raise HTTPException(status.HTTP_409_CONFLICT, NO_RULES)

    try:
        pool = pool_status(repo.get_grants(ctx.company_id), repo.get_pool_size(ctx.company_id))["unallocated"]
        board_date = repo.get_board_resolution_date(ctx.company_id)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / file_name
            path.write_bytes(data)
            result = checker(path, rules, pool, board_date)
    except Exception as exc:  # noqa: BLE001 - log it, audit it, return a clean 500
        logger.exception("compliance check failed for user %s", ctx.user_id)
        error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        try:
            audit(ctx, file_name, elapsed_ms(start), outcome="error", error=error)
        except Exception:  # noqa: BLE001 - the original error is what the client needs to hear about
            logger.exception("audit write failed after a compliance error")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, COMPLIANCE_ERROR) from None

    latency = elapsed_ms(start)
    flags = (["report_template_fallback"] if result.report.source == "template" else []) + \
            (["extraction_warnings"] if result.warnings else [])
    details = {"file_hash": result.file_hash, "letter_title": result.letter_title, "counts": result.counts,
               "findings": [{"id": f.id, "field": f.field, "status": f.status, "rule_ids": f.rule_ids}
                            for f in result.findings],
               "report_source": result.report.source, "report_problems": result.report.problems,
               "warnings": result.warnings, "llm_calls": result.llm_calls, "cached": result.cached,
               "pool_remaining": pool, "board_resolution_date": board_date.isoformat()}
    try:
        audit_id = audit(ctx, file_name, latency, outcome=result.outcome, summary=result.summary(),
                         details=details, flags=flags)
    except Exception:  # noqa: BLE001 - fail closed, as /chat
        logger.exception("audit write failed for user %s", ctx.user_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, COMPLIANCE_ERROR) from None
    response.headers["X-Audit-Id"] = audit_id
    return ComplianceResponse(letter_title=result.letter_title, file_hash=result.file_hash, outcome=result.outcome,
                              summary=result.summary(), counts=result.counts, findings=result.findings,
                              report=result.report, warnings=result.warnings, llm_calls=result.llm_calls,
                              cached=result.cached, latency_ms=latency)


# --- /audit ---

@app.get("/audit", response_model=list[AuditRecord], tags=["admin"])
def audit_log(
    ctx: AdminCtx,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
    reader: AuditReader = Depends(get_audit_reader),
) -> list[dict[str, Any]]:
    """Latest audit records for the admin's company, newest first (admin only)."""
    return reader(ctx.company_id, limit)
