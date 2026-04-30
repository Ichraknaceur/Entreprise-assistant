"""Tests for Langfuse-backed prompt management."""

from typing import TYPE_CHECKING, Literal, cast

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.core.prompt_management import (
    resolve_openai_chat_prompt,
)

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.prompt_management import (
        LangfusePromptClientProtocol,
    )


class FakePrompt:
    """Simple fake Langfuse prompt."""

    def __init__(self) -> None:
        """Initialize prompt metadata."""
        self.name = "enterprise-rag-answer-chat"
        self.version = 3
        self.is_fallback = False
        self.compile_calls: list[dict[str, object]] = []

    def compile(self, **kwargs: object) -> list[dict[str, object]]:
        """Record compile kwargs and return deterministic messages."""
        self.compile_calls.append(kwargs)
        return [
            {
                "role": "system",
                "content": "System from Langfuse",
            },
            {
                "role": "user",
                "content": "User from Langfuse",
            },
        ]


class FakeLangfuseClient:
    """Simple fake Langfuse client for prompt fetching."""

    def __init__(self, prompt: FakePrompt) -> None:
        """Store the fake prompt and call history."""
        self.prompt = prompt
        self.calls: list[dict[str, object]] = []

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        type: Literal["chat"] = "chat",  # noqa: A002
        fallback: list[dict[str, str]] | None = None,
    ) -> FakePrompt:
        """Record prompt fetch inputs and return the fake prompt."""
        self.calls.append(
            {
                "name": name,
                "label": label,
                "type": type,
                "fallback": fallback,
            },
        )
        return self.prompt


def test_resolve_openai_chat_prompt_uses_langfuse_prompt() -> None:
    """Prompt management should fetch and compile the Langfuse prompt."""
    prompt = FakePrompt()
    client = FakeLangfuseClient(prompt)
    settings = Settings(
        langfuse_openai_prompt_name="enterprise-rag-answer-chat",
        langfuse_prompt_label="production",
    )

    result = resolve_openai_chat_prompt(
        "What is the remote work policy?",
        ["Employees may work remotely up to three days."],
        settings=settings,
        client=cast("LangfusePromptClientProtocol", client),
    )

    assert result.name == "enterprise-rag-answer-chat"
    assert result.label == "production"
    assert result.version == 3
    assert result.is_fallback is False
    assert result.messages == [
        {"role": "system", "content": "System from Langfuse"},
        {"role": "user", "content": "User from Langfuse"},
    ]
    assert client.calls[0]["name"] == "enterprise-rag-answer-chat"
    assert client.calls[0]["label"] == "production"
    assert client.calls[0]["type"] == "chat"
    assert prompt.compile_calls == [
        {
            "question": "What is the remote work policy?",
            "contexts": "Context 1:\nEmployees may work remotely up to three days.",
        },
    ]


def test_resolve_openai_chat_prompt_returns_local_fallback_without_client() -> None:
    """Prompt management should fall back locally when Langfuse is unavailable."""
    result = resolve_openai_chat_prompt(
        "How do I access the VPN?",
        ["VPN access requires MFA and an approved device."],
        settings=Settings(
            langfuse_public_key=None,
            langfuse_secret_key=None,
        ),
        client=None,
    )

    assert result.is_fallback is True
    assert result.version is None
    assert result.name == "local-openai-fallback"
    assert len(result.messages) == 2
