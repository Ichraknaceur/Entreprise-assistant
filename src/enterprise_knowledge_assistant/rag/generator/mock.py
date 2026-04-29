"""Mock generator used before a real LLM provider is configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence


class MockGenerator(BaseGenerator):
    """Return a predictable placeholder response."""

    @property
    def provider_name(self) -> str:
        """Return the mock provider name."""
        return "mock"

    @property
    def model_name(self) -> str:
        """Return the mock model name."""
        return "mock-generator"

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate a placeholder answer for early development."""
        context_count = len(contexts)
        return (
            "Generation is scaffolded but not implemented yet. "
            f"Received question: '{question}'. "
            f"Retrieved context chunks: {context_count}."
        )
