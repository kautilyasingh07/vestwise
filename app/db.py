"""MongoDB client and collection handles (spec §7).

The client is created lazily on first use, so importing this module never
opens a network connection (useful for tests that don't need Mongo).
"""

from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi

from app.config import settings

SERVER_SELECTION_TIMEOUT_MS = 5000


@lru_cache
def get_client() -> MongoClient:
    """Return the process-wide MongoClient (it holds its own connection pool)."""
    return MongoClient(
        settings.mongo_uri.get_secret_value(),
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
    )


def get_db() -> Database:
    """Return the Vestwise database handle."""
    return get_client()[settings.mongo_db]


def ping() -> bool:
    """Round-trip to the server; raises a PyMongoError if it is unreachable."""
    get_client().admin.command("ping")
    return True


# One handle per collection in spec §7. Functions, not module constants,
# so the client is only created when a handle is first requested.
def companies() -> Collection:
    """Companies: name, total authorised shares."""
    return get_db()["companies"]


def users() -> Collection:
    """App users: role (employee/admin), company_id, stakeholder_id."""
    return get_db()["users"]


def stakeholders() -> Collection:
    """Founders, employees and investors of a company."""
    return get_db()["stakeholders"]


def holdings() -> Collection:
    """Issued shares per stakeholder and share class."""
    return get_db()["holdings"]


def grants() -> Collection:
    """Option grants with vesting schedule (options are not shares until exercised)."""
    return get_db()["grants"]


def documents() -> Collection:
    """Ingested PDFs, keyed on file hash."""
    return get_db()["documents"]


def chunks() -> Collection:
    """Document chunks with embeddings (Atlas vector index on `embedding`)."""
    return get_db()["chunks"]


def audit_logs() -> Collection:
    """One record per request: question, chunks, tool calls, answer, latency."""
    return get_db()["audit_logs"]
