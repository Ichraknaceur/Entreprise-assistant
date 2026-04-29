"""Tests for the OpenAI-backed generator."""

from typing import cast

from enterprise_knowledge_assistant.rag.generator.openai import OpenAIGenerator


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


def test_openai_generator_builds_responses_api_request() -> None:
    """OpenAI generator should send grounded prompts through the Responses API."""
    client = FakeOpenAIResponsesClient()
    generator = OpenAIGenerator(
        api_key="test-key",
        model_name="gpt-4o-mini",
        client=client,
    )

    result = generator.generate(
        question="What is the remote work policy?",
        contexts=["Employees may work remotely up to three days."],
    )

    assert result == "Grounded answer from OpenAI."
    assert client.calls[0]["model"] == "gpt-4o-mini"
    assert len(cast("list[dict[str, object]]", client.calls[0]["input_items"])) == 2
