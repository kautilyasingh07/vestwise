"""Ingest every PDF in data/docs/ for Nimbus Robotics (spec §10, FR-1..FR-5).

Run from the repo root:  python scripts/ingest_all.py

Only the top level of data/docs/ is ingested; data/docs/compliance/ holds
draft letters for the compliance checker (Phase 9), which must not become
searchable policy. Idempotent: running it twice leaves the same chunk count.

Grant letters are personal: each is ingested with its owner's stakeholder id
(checked against the seeded `stakeholders` collection); everything else is
company-wide (owner None).
"""

import sys
from pathlib import Path

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.ingest.loader import pdf_title  # noqa: E402
from app.ingest.pipeline import DocType, ingest_file  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
COMPANY_ID = "nimbus"  # data/seed.json company _id

# File-name prefix -> doc_type (spec §7).
DOC_TYPES: dict[str, DocType] = {
    "esop_policy": "policy",
    "grant_letter": "grant_letter",
    "board_resolution": "board_resolution",
}


# Personal documents -> owner stakeholder _id (data/seed.json). A grant letter missing
# from this map is refused by ingest_file rather than ingested as company-wide.
DOCUMENT_OWNERS: dict[str, str] = {
    "grant_letter_priya.pdf": "sh_priya",
    "grant_letter_rahul.pdf": "sh_rahul",
}


def check_owners_exist(company_id: str) -> None:
    """Raise if any DOCUMENT_OWNERS stakeholder is not seeded for this company."""
    seeded = {s["_id"] for s in db.stakeholders().find({"company_id": company_id}, {"_id": 1})}
    missing = set(DOCUMENT_OWNERS.values()) - seeded
    if missing:
        raise ValueError(f"owner stakeholders not in seed data for '{company_id}': {sorted(missing)}")


def doc_type_for(path: Path) -> DocType:
    """Infer doc_type from the file name; raise if no prefix matches."""
    for prefix, doc_type in DOC_TYPES.items():
        if path.stem.startswith(prefix):
            return doc_type
    raise ValueError(f"{path.name}: unknown document type (expected a prefix in {list(DOC_TYPES)})")


def main() -> int:
    """Ingest each top-level PDF and print per-file and total chunk counts."""
    db.ping()
    if db.companies().find_one({"_id": COMPANY_ID}) is None:
        print(f"Company '{COMPANY_ID}' not found; run scripts/seed.py first.")
        return 1
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))  # glob, not rglob: skips data/docs/compliance/
    if not pdfs:
        print(f"No PDFs in {DOCS_DIR}; run scripts/build_pdfs.py first.")
        return 1
    check_owners_exist(COMPANY_ID)
    for path in pdfs:
        owner = DOCUMENT_OWNERS.get(path.name)
        result = ingest_file(path, COMPANY_ID, doc_type_for(path), pdf_title(path), owner)
        print(f"{path.name}: {result.chunks} chunks from {result.pages} page(s), "
              f"replaced {result.replaced_chunks}, owner {result.owner_stakeholder_id or 'company-wide'} "
              f"(doc_id {result.doc_id})")
    total = db.chunks().count_documents({"company_id": COMPANY_ID})
    docs = db.documents().count_documents({"company_id": COMPANY_ID})
    print(f"total: {docs} documents, {total} chunks for company '{COMPANY_ID}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
