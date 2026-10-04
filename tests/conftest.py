"""Shared pytest fixtures."""

import json
from pathlib import Path
from typing import Any

import pytest

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seed.json"


@pytest.fixture(scope="session")
def seed() -> dict[str, Any]:
    """The parsed data/seed.json (dates still ISO strings)."""
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))
