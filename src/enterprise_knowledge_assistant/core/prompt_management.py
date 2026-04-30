"""Prompt management helpers backed by Langfuse with local fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.core.observability import get_langfuse_client
from enterprise_knowledge_assistant.rag.prompts import SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from enterprise_knowledge_assistant.core.config import Settings


class ChatPromptClientProtocol(Protocol):
    """Protocol for a Langfuse chat prompt client."""

    name: str
    version: int
    is_fallback: bool

    def compile(self, **kwargs: object) -> list[dict[str, str]]:
        """Compile the prompt with runtime variables."""


class LangfusePromptClientProtocol(Protocol):
    """Protocol for the Langfuse prompt client APIs used by this project."""

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        type: Literal["chat"] = "chat",  # noqa: A002
        fallback: list[dict[str, str]] | None = None,
    ) -> ChatPromptClientProtocol:
        """Retrieve a prompt by name and label."""


@dataclass(slots=True, frozen=True)
class ResolvedPrompt:
    """A prompt resolved from Langfuse or a local fallback."""

    name: str
    label: str | None
    version: int | None
    is_fallback: bool
    messages: list[dict[str, str]]


def resolve_openai_chat_prompt(
    question: str,
    contexts: Sequence[str],
    *,
    settings: Settings | None = None,
    client: LangfusePromptClientProtocol | None = None,
) -> ResolvedPrompt:
    """Resolve the OpenAI chat prompt from Langfuse with a local fallback."""
    resolved_settings = settings or get_settings()
    prompt_client = client or get_langfuse_client(resolved_settings)
    fallback_messages = _build_local_fallback_chat_prompt()
    context_text = _format_contexts(contexts)

    if prompt_client is None:
        return ResolvedPrompt(
            name="local-openai-fallback",
            label=None,
            version=None,
            is_fallback=True,
            messages=_compile_local_fallback_chat_prompt(
                question=question,
                contexts=contexts,
            ),
        )

    resolved_prompt_client = cast("LangfusePromptClientProtocol", prompt_client)
    prompt = resolved_prompt_client.get_prompt(
        resolved_settings.langfuse_openai_prompt_name,
        label=resolved_settings.langfuse_prompt_label,
        type="chat",
        fallback=fallback_messages,
    )
    compiled_messages = prompt.compile(
        question=question,
        contexts=context_text,
    )
    return ResolvedPrompt(
        name=prompt.name,
        label=resolved_settings.langfuse_prompt_label,
        version=prompt.version,
        is_fallback=prompt.is_fallback,
        messages=compiled_messages,
    )


def _build_local_fallback_chat_prompt() -> list[dict[str, str]]:
    """Return the local fallback chat prompt template."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Use only the context below to answer the user's question.\n\n"
                "Question:\n{{question}}\n\n"
                "Context:\n{{contexts}}"
            ),
        },
    ]


def _compile_local_fallback_chat_prompt(
    *,
    question: str,
    contexts: Sequence[str],
) -> list[dict[str, str]]:
    """Compile the local fallback prompt directly without Langfuse."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(question, contexts),
        },
    ]


def _format_contexts(contexts: Sequence[str]) -> str:
    """Format retrieved contexts into a single prompt variable."""
    return "\n\n".join(
        f"Context {index}:\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
