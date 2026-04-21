"""Factory for selecting an answer generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.rag.generator.mock import MockGenerator

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


def get_generator(settings: Settings) -> BaseGenerator:
    """Return the configured generator implementation."""
    if settings.llm_provider == "mock":
        return MockGenerator()
    msg = f"Unsupported LLM provider: {settings.llm_provider}"
    raise ValueError(msg)
