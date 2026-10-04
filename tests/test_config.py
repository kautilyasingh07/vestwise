"""Settings: LLM_MODEL is required (provider model names get retired, so no default)."""

import pytest

try:
    from pydantic import ValidationError

    from app.config import Settings
except Exception as exc:  # noqa: BLE001 - missing .env -> pydantic ValidationError at import
    pytest.skip(f"config unavailable: {exc}", allow_module_level=True)


def test_llm_model_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(_env_file=None, mongo_uri="mongodb://x")  # type: ignore[call-arg]


def test_llm_model_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    settings = Settings(_env_file=None, mongo_uri="mongodb://x")  # type: ignore[call-arg]
    assert settings.llm_model == "openai/gpt-oss-120b"
