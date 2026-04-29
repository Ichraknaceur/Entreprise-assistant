"""OpenAI-backed generator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from openai import OpenAI

from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator
from enterprise_knowledge_assistant.rag.prompts import SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence


class OpenAIResponseProtocol(Protocol):
    """Protocol for the minimal OpenAI response shape used by the app."""

    output_text: str


class OpenAIResponsesClientProtocol(Protocol):
    """Protocol for the OpenAI Responses API client used by this project."""

    def create(
        self,
        *,
        model: str,
        input: list[dict[str, object]],  # noqa: A002
    ) -> OpenAIResponseProtocol:
        """Create a model response."""


class OpenAIGenerator(BaseGenerator):
    """Generate grounded answers using the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        client: OpenAIResponsesClientProtocol | None = None,
    ) -> None:
        """Initialize the generator with an API key and model name."""
        self._model_name = model_name
        self._client = client or OpenAI(api_key=api_key).responses

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model name."""
        return self._model_name

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate an answer grounded in retrieved context."""
        response = self._client.create(
            model=self._model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_user_prompt(question, contexts),
                        },
                    ],
                },
            ],
        )
        return response.output_text.strip()
