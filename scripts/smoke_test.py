"""Phase 0 smoke test: can we reach MongoDB and the LLM?

Run from the repo root:  python scripts/smoke_test.py
Exits 0 only if both checks pass.
"""

import sys
from pathlib import Path

# `python scripts/x.py` puts scripts/ (not the repo root) on sys.path,
# so add the root to make `import app` work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db import get_db, ping  # noqa: E402
from app.llm import get_chat_model  # noqa: E402


def check_mongo() -> bool:
    """Ping the cluster and list collections in the Vestwise database."""
    try:
        ping()
        names = sorted(get_db().list_collection_names())
        print("[mongo] ping OK")
        print(f"[mongo] database '{settings.mongo_db}' collections: {names or '(none yet)'}")
        return True
    except Exception as exc:  # report any failure and keep going to the LLM check
        print(f"[mongo] FAILED: {type(exc).__name__}: {exc}")
        return False


def check_llm() -> bool:
    """Send a trivial prompt to the configured LLM and print its reply."""
    label = f"{settings.llm_provider}/{settings.resolved_llm_model}"
    try:
        reply = get_chat_model().invoke("Reply with OK")
        print(f"[llm] {label} replied: {reply.text.strip()!r}")
        return True
    except Exception as exc:
        print(f"[llm] {label} FAILED: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    """Run both checks; return a process exit code."""
    mongo_ok = check_mongo()
    llm_ok = check_llm()
    print("SMOKE TEST PASSED" if mongo_ok and llm_ok else "SMOKE TEST FAILED")
    return 0 if mongo_ok and llm_ok else 1


if __name__ == "__main__":
    sys.exit(main())
