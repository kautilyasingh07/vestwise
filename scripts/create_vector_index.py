"""Create the Atlas Vector Search index on `chunks` (spec §7, FR-4).

Run from the repo root:  python scripts/create_vector_index.py

Uses pymongo's search index API, then waits until Atlas reports the index
READY. If the API is refused (e.g. on a cluster tier that doesn't allow it),
prints the index JSON and the steps to create it in the Atlas UI instead.
Safe to re-run: an existing index with the same definition is left alone.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo.collection import Collection  # noqa: E402
from pymongo.errors import OperationFailure  # noqa: E402
from pymongo.operations import SearchIndexModel  # noqa: E402

from app import db  # noqa: E402
from app.config import settings  # noqa: E402

EMBEDDING_DIMS = 384  # all-MiniLM-L6-v2

# Spec §7 index definition: cosine similarity on `embedding`, pre-filter on `company_id`.
INDEX_DEFINITION: dict[str, Any] = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMS, "similarity": "cosine"},
        {"type": "filter", "path": "company_id"},
    ]
}

POLL_SECONDS = 5
TIMEOUT_SECONDS = 300


def find_index(collection: Collection, name: str) -> dict[str, Any] | None:
    """Return the search index with this name, or None."""
    return next(iter(collection.list_search_indexes(name)), None)


def ensure_index(collection: Collection, name: str) -> str:
    """Create the index, or update it if its definition differs; return what was done."""
    existing = find_index(collection, name)
    if existing is None:
        model = SearchIndexModel(definition=INDEX_DEFINITION, name=name, type="vectorSearch")
        collection.create_search_index(model)
        return "created"
    if existing.get("latestDefinition") != INDEX_DEFINITION:
        collection.update_search_index(name, INDEX_DEFINITION)
        return "updated"
    return "already exists"


def wait_until_ready(collection: Collection, name: str) -> str:
    """Poll until the index is READY and queryable (or FAILED / timed out); return the last status."""
    deadline = time.monotonic() + TIMEOUT_SECONDS
    status = "UNKNOWN"
    while time.monotonic() < deadline:
        index = find_index(collection, name) or {}
        status = index.get("status", "UNKNOWN")
        print(f"  status: {status}, queryable: {index.get('queryable', False)}")
        if (status == "READY" and index.get("queryable")) or status == "FAILED":
            return status
        time.sleep(POLL_SECONDS)
    return status


def print_manual_steps(name: str) -> None:
    """Print the index JSON and the Atlas UI steps to create it by hand."""
    print("\nCreate the index in the Atlas UI instead:")
    print("  1. Atlas -> your cluster -> Atlas Search tab (or 'Search & Vector Search').")
    print("  2. Create Search Index -> choose 'Vector Search' -> 'JSON Editor' -> Next.")
    print(f"  3. Database: {settings.mongo_db}, collection: chunks, index name: {name}")
    print("  4. Paste this definition, then Next -> Create Search Index:\n")
    print(json.dumps(INDEX_DEFINITION, indent=2))
    print("\n  5. Wait until the status shows READY (about a minute for a small collection).")


def main() -> int:
    """Create or verify the vector index and wait for it to become READY."""
    db.ping()
    collection = db.chunks()
    name = settings.vector_index_name
    if collection.estimated_document_count() == 0:
        print("chunks is empty; run scripts/ingest_all.py first (Atlas needs the collection to exist).")
        return 1
    try:
        print(f"index '{name}' on {settings.mongo_db}.chunks: {ensure_index(collection, name)}")
        status = wait_until_ready(collection, name)
    except OperationFailure as exc:
        print(f"Search index API failed: {exc}")
        print_manual_steps(name)
        return 1
    print(f"final status: {status}")
    return 0 if status == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
