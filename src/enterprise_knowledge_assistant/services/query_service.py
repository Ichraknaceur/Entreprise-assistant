"""Service layer for question-answering workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.api.schemas.query import QueryResponse, SourceItem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from enterprise_knowledge_assistant.api.schemas.query import QueryRequest
    from enterprise_knowledge_assistant.core.observability import (
        ObservabilityClientProtocol,
    )
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator
    from enterprise_knowledge_assistant.rag.retriever import RetrievedChunk


class QueryService:
    """Coordinate retrieval and answer generation for a query."""

    def __init__(
        self,
        *,
        default_provider: str,
        generator_factory: Callable[[str | None], BaseGenerator],
        retriever: Callable[[str], list[RetrievedChunk]],
        observability_client: ObservabilityClientProtocol,
    ) -> None:
        """Initialize the service with a retriever and answer generator factory."""
        self._default_provider = default_provider
        self._generator_factory = generator_factory
        self._retriever = retriever
        self._observability_client = observability_client

    def query(self, request: QueryRequest) -> QueryResponse:
        """Answer a query using retrieved source-backed context."""
        provider = request.provider or self._default_provider
        try:
            with self._observability_client.start_as_current_observation(
                name="query-request",
                as_type="span",
                input={
                    "question": request.question,
                    "provider": provider,
                },
            ) as query_observation:
                with self._observability_client.start_as_current_observation(
                    name="retrieve-context",
                    as_type="retriever",
                    input={"question": request.question},
                    metadata={"provider": provider},
                ) as retrieval_observation:
                    retrieved_chunks = self._retriever(request.question)
                    retrieval_observation.update(
                        output={
                            "result_count": len(retrieved_chunks),
                            "documents": [chunk.document for chunk in retrieved_chunks],
                        },
                    )

                if not retrieved_chunks:
                    response = QueryResponse(
                        answer=(
                            "I do not have enough information in the knowledge base "
                            "to answer that question."
                        ),
                        sources=[],
                    )
                    query_observation.update(
                        output={
                            "answer": response.answer,
                            "source_count": 0,
                            "refused": True,
                        },
                    )
                    return response

                generator = self._generator_factory(provider)
                contexts = _build_contexts(retrieved_chunks)
                with self._observability_client.start_as_current_observation(
                    name="generate-answer",
                    as_type="generation",
                    model=generator.model_name,
                    input={
                        "question": request.question,
                        "contexts": contexts,
                    },
                    metadata={
                        "provider": generator.provider_name,
                        "prompt_name": getattr(generator, "prompt_name", None),
                        "prompt_label": getattr(generator, "prompt_label", None),
                    },
                ) as generation_observation:
                    generated_answer = generator.generate(
                        question=request.question,
                        contexts=contexts,
                    )
                    generation_observation.update(output=generated_answer)

                response = QueryResponse(
                    answer=generated_answer,
                    sources=[_build_source_item(chunk) for chunk in retrieved_chunks],
                )
                query_observation.update(
                    output={
                        "answer": response.answer,
                        "source_count": len(response.sources),
                        "documents": [source.document for source in response.sources],
                    },
                )
                return response
        finally:
            self._observability_client.flush()


def _build_contexts(retrieved_chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Extract retrieval texts for answer generation."""
    return [chunk.text for chunk in retrieved_chunks]


def _build_source_item(chunk: RetrievedChunk) -> SourceItem:
    """Convert a retrieved chunk into the API response schema."""
    return SourceItem(document=chunk.document, snippet=chunk.text)
