"""Application settings, loaded from environment variables and `.env`.

This is the only module that reads the environment. Everything else does
`from app.config import settings`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "groq"]
RetrievalMode = Literal["vector", "hybrid"]

# `.env` at the repo root, not the working directory: an MCP client (Claude Desktop)
# starts the server from a directory of its choosing.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Typed view of the environment. Field names map to env vars case-insensitively."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB
    mongo_uri: SecretStr
    mongo_db: str = "vestwise"

    # LLM
    llm_provider: LLMProvider = "gemini"
    # Required, no default: provider model names get retired (gemini-2.5-flash now 404s),
    # so a hard-coded fallback fails late and confusingly. Missing -> error at startup.
    llm_model: str
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Embeddings and retrieval (spec §8.2)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_name: str = "chunks_vector_index"  # Atlas Vector Search index on chunks.embedding
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.35
    # "vector" (default) = $vectorSearch only; "hybrid" adds BM25 merged by RRF. Phase 10 measured
    # no gain on the golden set (hit@5 10/11 both), so hybrid is opt-in.
    retrieval_mode: RetrievalMode = "vector"

    # Domain (spec §8.4): must match the exercise window in the ESOP policy
    exercise_window_days: int = 90

    # UI -> API
    api_url: str = "http://localhost:8000"

    # MCP server (FR-20): the one user a server process acts for; no default, so a
    # missing value refuses to start instead of falling back to some user.
    vestwise_user_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (fails fast if a required variable is missing)."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment


settings = get_settings()
