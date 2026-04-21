"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "Enterprise Knowledge Assistant"
    app_version: str = "0.1.0"
    app_description: str = (
        "A RAG-based assistant for querying internal enterprise documentation."
    )
    api_prefix: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True

    data_dir: Path = Field(default=Path("data/sample_docs"))
    vector_store_dir: Path = Field(default=Path("storage/chroma"))

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "mock"
    llm_model_name: str = "mock-generator"
    openai_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
