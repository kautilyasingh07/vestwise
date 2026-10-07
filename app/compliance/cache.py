"""Cache of live LLM outputs, one JSON file per input file: data/compliance_cache/<file_hash>.json.

The free Gemini tier allows ~20 requests a day, so every live LLM output of the
checker (letter extraction, report text, the policy-rules proposal) is written
here and reused on reruns unless the caller asks to refresh. Keyed on the
SHA-256 of the input file's bytes: the same PDF always maps to the same entry,
whatever it is called.

The cache also makes the pipeline testable: a cached extraction *is* a recorded
LLM output, so tests replay real model answers without calling the model.

Entry layout (only the keys that have been produced are present):
    {"file_hash", "file_name",
     "extraction": {"model", "prompt_version", "created_at", "terms": {...}},
     "reports": {<findings_hash>: {"model", "prompt_version", "created_at", "text"}},
     "rules_proposal": {...}}
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "compliance_cache"


def file_hash(path: Path) -> str:
    """SHA-256 of the file's bytes, as hex (same as ingestion's, without importing the embedder)."""
    return bytes_hash(path.read_bytes())


def bytes_hash(data: bytes) -> str:
    """SHA-256 of some bytes, as hex."""
    return hashlib.sha256(data).hexdigest()


def json_hash(data: Any) -> str:
    """SHA-256 of a JSON-serialisable value in canonical form (sorted keys), as hex."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def entry_path(sha256: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Where the entry for one file hash lives."""
    return cache_dir / f"{sha256}.json"


def load_entry(sha256: str, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    """The cached entry for a file hash, or {} if there is none."""
    path = entry_path(sha256, cache_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_entry(sha256: str, entry: dict[str, Any], cache_dir: Path = CACHE_DIR) -> Path:
    """Write the entry (pretty-printed, so a human can read and diff it) and return its path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = entry_path(sha256, cache_dir)
    path.write_text(json.dumps(entry, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def now_iso() -> str:
    """Current UTC time as ISO 8601, for created_at stamps."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def is_quota_error(exc: BaseException) -> bool:
    """True for a rate-limit / quota error (HTTP 429, Gemini RESOURCE_EXHAUSTED) from any provider SDK."""
    if getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429:
        return True
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in ("429", "RESOURCE_EXHAUSTED", "RateLimit", "rate limit", "quota"))
