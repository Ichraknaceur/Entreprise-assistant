"""Prompt templates for grounded answer generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

SYSTEM_PROMPT = """
You are an internal enterprise knowledge assistant.
Answer only from the provided context and cite supporting sources.
If the context is insufficient, say you do not have enough information.
""".strip()


def build_user_prompt(question: str, contexts: Sequence[str]) -> str:
    """Build the grounded user prompt sent to the LLM provider."""
    formatted_contexts = "\n\n".join(
        f"Context {index}:\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
    return (
        "Use only the context below to answer the user's question.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{formatted_contexts}"
    )
