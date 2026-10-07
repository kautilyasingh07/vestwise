"""Load data/seed.json into MongoDB (spec §7).

Run from the repo root:  python scripts/seed.py

Idempotent: each seeded collection is dropped and reloaded, so running it
twice leaves exactly the same data. Also creates the spec §7 indexes on all
collections (not the Atlas vector index, which is created in the Atlas UI).
`documents`, `chunks` and `audit_logs` are never dropped here.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import ASCENDING, DESCENDING  # noqa: E402
from pymongo.database import Database  # noqa: E402

from app.db import get_db, ping  # noqa: E402

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seed.json"

# seed.json key -> Mongo collection name.
SEEDED_COLLECTIONS: dict[str, str] = {
    "company": "companies",
    "users": "users",
    "stakeholders": "stakeholders",
    "holdings": "holdings",
    "grants": "grants",
}

# Fields stored as BSON dates rather than "YYYY-MM-DD" strings.
DATE_FIELDS = ("grant_date", "termination_date", "esop_board_resolution_date")

# Spec §7 indexes: collection -> list of (keys, options).
INDEXES: dict[str, list[tuple[list[tuple[str, int]], dict[str, Any]]]] = {
    "users": [([("company_id", ASCENDING)], {})],
    "stakeholders": [([("company_id", ASCENDING)], {})],
    "holdings": [([("company_id", ASCENDING)], {})],
    "grants": [([("company_id", ASCENDING), ("stakeholder_id", ASCENDING)], {})],
    "documents": [([("company_id", ASCENDING), ("file_hash", ASCENDING)], {"unique": True})],
    "audit_logs": [([("company_id", ASCENDING), ("ts", DESCENDING)], {})],
}


def load_seed(path: Path = SEED_PATH) -> dict[str, Any]:
    """Read seed.json as a plain dict (dates still strings)."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dates(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of doc with DATE_FIELDS converted from ISO strings to datetimes."""
    out = dict(doc)
    for name in DATE_FIELDS:
        if out.get(name):
            out[name] = datetime.fromisoformat(out[name])
    return out


def to_mongo_docs(seed: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map seed.json to {collection_name: [documents]}. Pure: no database access."""
    result: dict[str, list[dict[str, Any]]] = {}
    for key, collection in SEEDED_COLLECTIONS.items():
        value = seed[key]
        docs = value if isinstance(value, list) else [value]  # "company" is a single object
        result[collection] = [parse_dates(d) for d in docs]
    return result


def reload_collection(db: Database, name: str, docs: list[dict[str, Any]]) -> int:
    """Drop a collection and insert docs; return the resulting document count."""
    db.drop_collection(name)
    if docs:
        db[name].insert_many(docs)
    return db[name].count_documents({})


def create_indexes(db: Database) -> list[str]:
    """Create the spec §7 indexes (no-op if they already exist); return their names."""
    names = []
    for collection, specs in INDEXES.items():
        for keys, options in specs:
            names.append(f"{collection}.{db[collection].create_index(keys, **options)}")
    return names


def main() -> int:
    """Reload every seeded collection, create indexes, and print counts."""
    ping()
    db = get_db()
    for collection, docs in to_mongo_docs(load_seed()).items():
        print(f"{collection}: {reload_collection(db, collection, docs)} documents")
    for name in create_indexes(db):
        print(f"index {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
