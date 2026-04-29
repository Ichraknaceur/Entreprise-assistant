"""Factory for selecting an answer generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.rag.generator.mock import MockGenerator
from enterprise_knowledge_assistant.rag.generator.openai import OpenAIGenerator

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


def get_generator(
    settings: Settings,
    *,
    provider: str | None = None,
) -> BaseGenerator:
    """Return the configured generator implementation."""
    resolved_provider = provider or settings.llm_provider
    if resolved_provider == "mock":
        return MockGenerator()
    if resolved_provider == "openai":
        if not settings.openai_api_key:
            msg = "OpenAI API key is not configured."
            raise ValueError(msg)
        return OpenAIGenerator(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_name,
        )
    msg = f"Unsupported LLM provider: {resolved_provider}"
    raise ValueError(msg)
