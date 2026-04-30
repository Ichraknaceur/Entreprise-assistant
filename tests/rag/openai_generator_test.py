"""Tests for the OpenAI-backed generator."""

from typing import TYPE_CHECKING, Literal, cast

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.rag.generator.openai import OpenAIGenerator

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.prompt_management import (
        LangfusePromptClientProtocol,
    )
    from enterprise_knowledge_assistant.rag.generator.openai import (
        OpenAIResponsesClientProtocol,
    )


class FakeOpenAIResponse:
    """Simple fake response object exposing output_text."""

    def __init__(self, output_text: str) -> None:
        """Store the configured model output."""
        self.output_text = output_text


class FakeOpenAIResponsesClient:
    """Simple fake OpenAI Responses API client."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        input: list[dict[str, object]],  # noqa: A002
    ) -> FakeOpenAIResponse:
        """Record the request and return a deterministic response."""
        self.calls.append({"model": model, "input_items": input})
        return FakeOpenAIResponse("Grounded answer from OpenAI.  ")


class FakePromptClient:
    """Simple fake Langfuse prompt client."""

    def __init__(self) -> None:
        """Initialize compile call storage."""
        self.compile_calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        """Return the prompt name."""
        return "enterprise-rag-answer-chat"

    @property
    def version(self) -> int:
        """Return the prompt version."""
        return 2

    @property
    def is_fallback(self) -> bool:
        """Return the fallback status."""
        return False

    def compile(self, **kwargs: object) -> list[dict[str, object]]:
        """Record compile kwargs and return prompt messages."""
        self.compile_calls.append(kwargs)
        return [
            {
                "role": "system",
                "content": "System prompt from Langfuse",
            },
            {
                "role": "user",
                "content": "User prompt from Langfuse",
            },
        ]


class FakeLangfuseClient:
    """Simple fake Langfuse client for OpenAI generator tests."""

    def __init__(self, prompt: FakePromptClient) -> None:
        """Store the prompt and call history."""
        self.prompt = prompt
        self.calls: list[dict[str, object]] = []

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        type: Literal["chat"] = "chat",  # noqa: A002
        fallback: list[dict[str, str]] | None = None,
    ) -> FakePromptClient:
        """Record fetch inputs and return the fake prompt."""
        self.calls.append(
            {
                "name": name,
                "label": label,
                "type": type,
                "fallback": fallback,
            },
        )
        return self.prompt


def test_openai_generator_builds_responses_api_request() -> None:
    """OpenAI generator should send grounded prompts through the Responses API."""
    response_client = FakeOpenAIResponsesClient()
    prompt_client = FakePromptClient()
    langfuse_client = FakeLangfuseClient(prompt_client)
    generator = OpenAIGenerator(
        api_key="test-key",
        model_name="gpt-4o-mini",
        settings=Settings(
            langfuse_openai_prompt_name="enterprise-rag-answer-chat",
            langfuse_prompt_label="production",
        ),
        client=cast("OpenAIResponsesClientProtocol", response_client),
        langfuse_client=cast("LangfusePromptClientProtocol", langfuse_client),
    )

    result = generator.generate(
        question="What is the remote work policy?",
        contexts=["Employees may work remotely up to three days."],
    )

    assert result == "Grounded answer from OpenAI."
    assert response_client.calls[0]["model"] == "gpt-4o-mini"
    assert (
        len(
            cast("list[dict[str, object]]", response_client.calls[0]["input_items"]),
        )
        == 2
    )
    assert langfuse_client.calls[0]["name"] == "enterprise-rag-answer-chat"
    assert prompt_client.compile_calls == [
        {
            "question": "What is the remote work policy?",
            "contexts": "Context 1:\nEmployees may work remotely up to three days.",
        },
    ]
