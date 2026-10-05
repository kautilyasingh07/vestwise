"""Request and response models for the API (spec §9).

Request models carry **no identity fields**: who is asking comes only from the
X-User-Id header (app/main.py). Unknown body fields are ignored (pydantic's
default `extra="ignore"`), so a body that sends `user_id`, `role`,
`company_id` or `stakeholder_id` has no effect.
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.audit import Outcome

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY = 20  # accepted; the agent keeps only the last 6 (FR-16)


# --- /chat ---

class HistoryItem(BaseModel):
    """One earlier turn of the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_CHARS * 2)


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[HistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY)
    as_of: date | None = Field(default=None, description="Override 'today' (demo/eval); default is the real date")


class Citation(BaseModel):
    """A source the answer cites: one page of one document."""

    doc_title: str
    page: int
    section: str
    snippet: str
    chunk_id: str


class ToolCall(BaseModel):
    """A tool the agent ran, with the arguments the model chose."""

    name: str
    args: dict[str, Any]


class ChatResponse(BaseModel):
    """Answer plus everything needed to show and check it."""

    answer: str
    outcome: Outcome = Field(description="answered | refused | not_found | error (same classifier as the audit log)")
    citations: list[Citation]
    tool_calls: list[ToolCall]
    latency_ms: int


# --- /vesting ---

class GrantVesting(BaseModel):
    """Vesting status of one grant on `as_of` (spec §8.4)."""

    grant_id: str | None
    as_of: date
    granted: int
    vested: int
    unvested: int
    exercised: int
    exercisable: int
    lapsed: int
    months_elapsed: int
    next_vest_date: date | None
    next_vest_options: int
    is_terminated: bool
    exercise_deadline: date | None


class VestingResponse(BaseModel):
    """All grants of one stakeholder."""

    stakeholder_id: str
    name: str
    as_of: date
    grants: list[GrantVesting]


# --- /captable ---

class CapTableRow(BaseModel):
    """One holder (or the unallocated pool, stakeholder_id None)."""

    stakeholder_id: str | None
    name: str
    shares: int
    outstanding_options: int
    fully_diluted: int
    issued_pct: float
    fully_diluted_pct: float


class PoolStatus(BaseModel):
    """ESOP pool accounting (spec §7)."""

    pool_size: int
    outstanding: int
    exercised: int
    lapsed: int
    unallocated: int


class CapTableResponse(BaseModel):
    """Issued and fully diluted ownership."""

    rows: list[CapTableRow]
    total_issued: int
    total_fully_diluted: int
    pool: PoolStatus


class SimulateRequest(BaseModel):
    """Body of POST /captable/simulate."""

    new_shares: int = Field(gt=0)
    investor_name: str = Field(min_length=1, max_length=200)


class DilutionRow(BaseModel):
    """Before/after ownership of one holder."""

    stakeholder_id: str | None
    name: str
    shares: int
    outstanding_options: int
    issued_pct_before: float
    issued_pct_after: float
    fully_diluted_pct_before: float
    fully_diluted_pct_after: float


class DilutionResponse(BaseModel):
    """Result of a dilution scenario (spec §8.5)."""

    investor_name: str
    new_shares: int
    rows: list[DilutionRow]
    total_issued_before: int
    total_issued_after: int
    total_fully_diluted_before: int
    total_fully_diluted_after: int


# --- /documents ---

class DocumentResponse(BaseModel):
    """What POST /documents ingested."""

    doc_id: str
    chunks_created: int
    title: str
    pages: int
    owner_stakeholder_id: str | None
    replaced_chunks: int


# --- /audit ---

class AuditRecord(BaseModel):
    """One audit_logs entry (FR-18)."""

    id: str
    ts: datetime
    user_id: str
    role: str
    stakeholder_id: str | None
    question: str
    as_of: date | None
    model: str
    chunk_ids: list[str]
    tool_calls: list[ToolCall]
    answer: str | None
    outcome: str
    error: str | None
    flags: list[str] = Field(default_factory=list)  # e.g. citation_retried, citation_invalid
    citation_check: dict[str, Any] | None = None  # first draft: total/invalid in-text citations; retried; stripped
    latency_ms: int
