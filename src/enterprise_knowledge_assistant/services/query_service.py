"""Service layer for question-answering workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.api.schemas.query import QueryResponse, SourceItem

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.api.schemas.query import QueryRequest
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


class QueryService:
    """Coordinate retrieval and answer generation for a query."""

    def __init__(self, generator: BaseGenerator) -> None:
        """Initialize the service with an answer generator."""
        self._generator = generator

    def query(self, request: QueryRequest) -> QueryResponse:
        """Answer a query using the configured generator placeholder."""
        generated_answer = self._generator.generate(
            question=request.question,
            contexts=[],
        )
        return QueryResponse(
            answer=generated_answer,
            sources=[
                SourceItem(
                    document="not_implemented_yet",
                    snippet=(
                        "Retrieval pipeline will provide source-backed snippets here."
                    ),
                ),
            ],
        )
