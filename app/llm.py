"""LLM factory: returns a LangChain chat model for the configured provider.

Callers depend only on `BaseChatModel`, so swapping Gemini for Groq is a
change to LLM_PROVIDER in `.env`, not to code (spec §5, Swappability).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from app.config import settings

# Fixed for every provider: demo and eval runs should be as repeatable as possible.
# Not a parameter, so no caller can quietly raise it. Note: ChatGroq sends 0 as 1e-8
# (Groq's API special-cases exactly 0); that's still effectively greedy decoding.
TEMPERATURE = 0.0


def _require_key(key: SecretStr | None, env_name: str) -> SecretStr:
    """Return the key or raise a clear error naming the missing variable."""
    if key is None or not key.get_secret_value():
        raise RuntimeError(f"{env_name} is not set but LLM_PROVIDER={settings.llm_provider}")
    return key


def get_chat_model(max_retries: int | None = None) -> BaseChatModel:
    """Build the chat model selected by LLM_PROVIDER and LLM_MODEL, at TEMPERATURE (0).

    Temperature 0 makes tool choices and wording as repeatable as possible; it does
    not guarantee identical output (see learning/phase-06-api.md).
    `max_retries=None` keeps the provider client's own retry-with-backoff on errors such
    as HTTP 429. The eval passes 0 so it can do (and time) the backoff itself.
    """
    retries = {} if max_retries is None else {"max_retries": max_retries}
    model = settings.llm_model

    # Provider imports are local so only the chosen SDK is loaded.
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_require_key(settings.google_api_key, "GOOGLE_API_KEY"),
            temperature=TEMPERATURE,
            **retries,
        )

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=_require_key(settings.groq_api_key, "GROQ_API_KEY"),
            temperature=TEMPERATURE,
            **retries,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
