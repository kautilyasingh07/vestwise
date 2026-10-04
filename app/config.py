"""Application settings, loaded from environment variables and `.env`.

This is the only module that reads the environment. Everything else does
`from app.config import settings`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "groq"]

# Used when LLM_MODEL is not set, so switching provider is a one-line change.
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}


class Settings(BaseSettings):
    """Typed view of the environment. Field names map to env vars case-insensitively."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB
    mongo_uri: SecretStr
    mongo_db: str = "vestwise"

    # LLM
    llm_provider: LLMProvider = "gemini"
    llm_model: str | None = None
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Embeddings and retrieval (spec §8.2)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_name: str = "chunks_vector_index"  # Atlas Vector Search index on chunks.embedding
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.35

    # Domain (spec §8.4): must match the exercise window in the ESOP policy
    exercise_window_days: int = 90

    # UI -> API
    api_url: str = "http://localhost:8000"

    @property
    def resolved_llm_model(self) -> str:
        """Return LLM_MODEL if set, else the default model for the chosen provider."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (fails fast if a required variable is missing)."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment


settings = get_settings()
