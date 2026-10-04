"""Ingest one PDF into `documents` and `chunks` (spec FR-1..FR-5, §7).

Idempotent (FR-5): the document id is derived from the company and the file's
SHA-256, so re-ingesting the same bytes deletes that document's old chunks
and writes the same ids again instead of adding duplicates.

Document-level access: every document and chunk carries `owner_stakeholder_id`,
None for company-wide documents (policy, board resolution) and the employee's
stakeholder id for personal ones (grant letters). The retriever filters on it
(app/rag/access.py), so employees never retrieve another employee's letter.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app import db
from app.ingest.chunker import Chunk, chunk_document
from app.ingest.embedder import embed
from app.ingest.loader import load_pdf

DocType = Literal["policy", "grant_letter", "board_resolution"]

# Doc types that belong to one stakeholder; ingesting one without an owner would make it company-wide.
PERSONAL_DOC_TYPES: frozenset[str] = frozenset({"grant_letter"})

HASH_PREFIX_LEN = 12  # hex chars of the SHA-256 used in ids (48 bits: no collisions at this scale)
READ_BLOCK = 1 << 16


@dataclass(frozen=True)
class IngestResult:
    """What ingest_file wrote, for logging."""

    doc_id: str
    title: str
    owner_stakeholder_id: str | None
    pages: int
    chunks: int
    replaced_chunks: int


def file_hash(path: Path) -> str:
    """SHA-256 of the file's bytes, as hex. Same bytes -> same hash, whatever the file name."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(READ_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def make_doc_id(company_id: str, sha256: str) -> str:
    """Deterministic document id: '<company_id>_<first 12 hex chars of the hash>'."""
    return f"{company_id}_{sha256[:HASH_PREFIX_LEN]}"


def check_owner(doc_type: DocType, owner_stakeholder_id: str | None) -> None:
    """Fail closed: a personal document (grant letter) must name its owner."""
    if doc_type in PERSONAL_DOC_TYPES and not owner_stakeholder_id:
        raise ValueError(f"{doc_type} documents need an owner_stakeholder_id; without one every employee could read it")


def chunk_records(
    chunks: list[Chunk],
    vectors: list[list[float]],
    doc_id: str,
    company_id: str,
    title: str,
    doc_type: DocType,
    owner_stakeholder_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build the Mongo `chunks` documents (spec §7 + FR-3 metadata). Pure: no I/O."""
    return [
        {
            "_id": f"{doc_id}_{i:03d}",
            "company_id": company_id,
            "owner_stakeholder_id": owner_stakeholder_id,
            "doc_id": doc_id,
            "doc_title": title,
            "doc_type": doc_type,
            "page": chunk.page,
            "section": chunk.section,
            "chunk_index": i,
            "text": chunk.text,
            "embedding": vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]


def ingest_file(
    path: Path, company_id: str, doc_type: DocType, title: str, owner_stakeholder_id: str | None = None
) -> IngestResult:
    """Load, chunk, embed and store one PDF; replace any earlier ingestion of the same file.

    owner_stakeholder_id is None for company-wide documents and required for
    grant letters. All slow work (parsing, embedding) happens before the first
    write, so a failure there leaves the database untouched. The writes are not
    a transaction; if a run dies mid-write, running it again repairs the state.
    """
    check_owner(doc_type, owner_stakeholder_id)
    sha256 = file_hash(path)
    doc_id = make_doc_id(company_id, sha256)
    pages = load_pdf(path)
    chunks = chunk_document(pages, title)
    vectors = embed([c.embed_text for c in chunks])
    records = chunk_records(chunks, vectors, doc_id, company_id, title, doc_type, owner_stakeholder_id)

    replaced = db.chunks().delete_many({"company_id": company_id, "doc_id": doc_id}).deleted_count
    db.documents().replace_one(
        {"company_id": company_id, "file_hash": sha256},
        {
            "_id": doc_id,
            "company_id": company_id,
            "owner_stakeholder_id": owner_stakeholder_id,
            "title": title,
            "doc_type": doc_type,
            "file_hash": sha256,
            "filename": path.name,
            "pages": len(pages),
            "ingested_at": datetime.now(UTC),
        },
        upsert=True,
    )
    if records:
        db.chunks().insert_many(records)
    return IngestResult(
        doc_id=doc_id,
        title=title,
        owner_stakeholder_id=owner_stakeholder_id,
        pages=len(pages),
        chunks=len(records),
        replaced_chunks=replaced,
    )
