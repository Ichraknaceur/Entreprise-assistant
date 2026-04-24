"""Service layer for question-answering workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.api.schemas.query import QueryResponse, SourceItem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from enterprise_knowledge_assistant.api.schemas.query import QueryRequest
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator
    from enterprise_knowledge_assistant.rag.retriever import RetrievedChunk


class QueryService:
    """Coordinate retrieval and answer generation for a query."""

    def __init__(
        self,
        *,
        generator: BaseGenerator,
        retriever: Callable[[str], list[RetrievedChunk]],
    ) -> None:
        """Initialize the service with a retriever and answer generator."""
        self._generator = generator
        self._retriever = retriever

    def query(self, request: QueryRequest) -> QueryResponse:
        """Answer a query using retrieved source-backed context."""
        retrieved_chunks = self._retriever(request.question)
        generated_answer = self._generator.generate(
            question=request.question,
            contexts=_build_contexts(retrieved_chunks),
        )
        return QueryResponse(
            answer=generated_answer,
            sources=[_build_source_item(chunk) for chunk in retrieved_chunks],
        )


def _build_contexts(retrieved_chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Extract retrieval texts for answer generation."""
    return [chunk.text for chunk in retrieved_chunks]


def _build_source_item(chunk: RetrievedChunk) -> SourceItem:
    """Convert a retrieved chunk into the API response schema."""
    return SourceItem(document=chunk.document, snippet=chunk.text)
