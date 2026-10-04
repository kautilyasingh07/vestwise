"""LLM factory: returns a LangChain chat model for the configured provider.

Callers depend only on `BaseChatModel`, so swapping Gemini for Groq is a
change to LLM_PROVIDER in `.env`, not to code (spec §5, Swappability).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from app.config import settings


def _require_key(key: SecretStr | None, env_name: str) -> SecretStr:
    """Return the key or raise a clear error naming the missing variable."""
    if key is None or not key.get_secret_value():
        raise RuntimeError(f"{env_name} is not set but LLM_PROVIDER={settings.llm_provider}")
    return key


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Build the chat model selected by LLM_PROVIDER.

    Temperature defaults to 0 so answers and tool choices are as repeatable as possible.
    """
    model = settings.llm_model

    # Provider imports are local so only the chosen SDK is loaded.
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_require_key(settings.google_api_key, "GOOGLE_API_KEY"),
            temperature=temperature,
        )

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=_require_key(settings.groq_api_key, "GROQ_API_KEY"),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
