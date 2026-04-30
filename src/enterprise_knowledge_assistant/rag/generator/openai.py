"""OpenAI-backed generator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from openai import OpenAI

from enterprise_knowledge_assistant.core.prompt_management import (
    resolve_openai_chat_prompt,
)
from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai.types.responses.response_input_param import ResponseInputParam

    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.core.prompt_management import (
        LangfusePromptClientProtocol,
    )


class OpenAIResponseProtocol(Protocol):
    """Protocol for the minimal OpenAI response shape used by the app."""

    output_text: str


class OpenAIResponsesClientProtocol(Protocol):
    """Protocol for the OpenAI Responses API client used by this project."""

    def create(
        self,
        *,
        model: str,
        input: object,  # noqa: A002
    ) -> OpenAIResponseProtocol:
        """Create a model response."""


class OpenAIGenerator(BaseGenerator):
    """Generate grounded answers using the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        settings: Settings | None = None,
        client: OpenAIResponsesClientProtocol | None = None,
        langfuse_client: LangfusePromptClientProtocol | None = None,
    ) -> None:
        """Initialize the generator with an API key and model name."""
        self._model_name = model_name
        self._settings = settings
        self._client = client or OpenAI(api_key=api_key).responses
        self._langfuse_client = langfuse_client

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model name."""
        return self._model_name

    @property
    def prompt_name(self) -> str | None:
        """Return the configured Langfuse prompt name."""
        return self._settings.langfuse_openai_prompt_name if self._settings else None

    @property
    def prompt_label(self) -> str | None:
        """Return the configured Langfuse prompt label."""
        return self._settings.langfuse_prompt_label if self._settings else None

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate an answer grounded in retrieved context."""
        resolved_prompt = resolve_openai_chat_prompt(
            question,
            contexts,
            settings=self._settings,
            client=self._langfuse_client,
        )
        response = self._client.create(
            model=self._model_name,
            input=cast("ResponseInputParam", resolved_prompt.messages),
        )
        return response.output_text.strip()
