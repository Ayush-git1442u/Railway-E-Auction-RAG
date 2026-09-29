"""Centralised, environment-driven application settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`.

    Every field can be overridden by an environment variable of the same
    name (case-insensitive), e.g. ``GROQ_API_KEY`` or ``RETRIEVAL_K``.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Groq / LLM ---------------------------------------------------------
    groq_api_key: SecretStr = Field(..., description="Groq API key (required).")
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = Field(0.5, ge=0.0, le=2.0)
    llm_max_completion_tokens: int = Field(1024, gt=0)
    llm_top_p: float = Field(1.0, gt=0.0, le=1.0)
    llm_reasoning_effort: Literal["low", "medium", "high"] = "medium"
    llm_timeout_seconds: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(2, ge=0)

    # --- Embeddings / Vector store -----------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_path: Path = BASE_DIR / "vectorstore" / "db_faiss"
    retrieval_k: int = Field(3, gt=0)
    # Required by LangChain to unpickle the docstore. Only enable for indexes
    # you built yourself and fully trust.
    faiss_allow_dangerous_deserialization: bool = True

    # --- Pipeline / Logging -------------------------------------------------
    chat_history_limit: int = Field(50, ge=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()  # type: ignore[call-arg]
