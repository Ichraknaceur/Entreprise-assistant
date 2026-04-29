"""Retrieval helpers for Milvus-backed semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.rag.embeddings import build_query_embedding
from enterprise_knowledge_assistant.rag.vector_store import get_vector_store

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.rag.embeddings import EmbeddingModelProtocol


class MilvusSearchClientProtocol(Protocol):
    """Protocol for the subset of Milvus search APIs used by retrieval."""

    def search(
        self,
        collection_name: str,
        *,
        data: list[list[float]],
        limit: int,
        output_fields: list[str],
        search_params: dict[str, str],
    ) -> list[list[dict[str, Any]]]:
        """Search a collection and return matched entities."""


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search."""

    chunk_id: str
    document: str
    category: str
    path: str
    title: str
    text: str
    score: float


def retrieve_context(
    question: str,
    *,
    limit: int = 3,
    settings: Settings | None = None,
    model: EmbeddingModelProtocol | None = None,
    client: MilvusSearchClientProtocol | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a user query."""
    resolved_settings = settings or get_settings()
    query_embedding = build_query_embedding(
        question,
        settings=resolved_settings,
        model=model,
    )
    resolved_client = cast(
        "MilvusSearchClientProtocol",
        client or get_vector_store(resolved_settings),
    )
    search_results = resolved_client.search(
        resolved_settings.milvus_collection_name,
        data=[query_embedding],
        limit=limit,
        output_fields=["document", "category", "path", "title", "text"],
        search_params={"metric_type": "COSINE"},
    )
    if not search_results:
        return []

    return [_build_retrieved_chunk(match) for match in search_results[0]]


def _build_retrieved_chunk(match: dict[str, Any]) -> RetrievedChunk:
    """Normalize one Milvus search match into the app retrieval model."""
    entity = cast("dict[str, Any]", match.get("entity", match))
    return RetrievedChunk(
        chunk_id=str(entity.get("id", match.get("id", ""))),
        document=str(entity["document"]),
        category=str(entity["category"]),
        path=str(entity["path"]),
        title=str(entity["title"]),
        text=str(entity["text"]),
        score=float(match.get("distance", match.get("score", 0.0))),
    )
