"""Document ingestion pipeline for Milvus-backed chunk storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.rag.vector_store import (
    MilvusCollectionClientProtocol,
    ensure_milvus_collection,
    get_vector_store,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.rag.chunking import KnowledgeChunk


class MilvusIngestionClientProtocol(MilvusCollectionClientProtocol, Protocol):
    """Protocol for the subset of Milvus APIs used during ingestion."""

    def upsert(self, *, collection_name: str, data: list[dict[str, Any]]) -> None:
        """Upsert records into a collection."""


@dataclass(slots=True, frozen=True)
class ChunkEmbeddingRecord:
    """A chunk bundled with its embedding vector for storage."""

    chunk: KnowledgeChunk
    embedding: list[float]


def ingest_documents(
    records: Sequence[ChunkEmbeddingRecord],
    settings: Settings | None = None,
    client: MilvusIngestionClientProtocol | None = None,
) -> dict[str, int]:
    """Ensure the collection exists and insert embedded chunks into Milvus."""
    resolved_settings = settings or get_settings()
    resolved_client = cast(
        "MilvusIngestionClientProtocol",
        client or get_vector_store(resolved_settings),
    )
    ensure_milvus_collection(resolved_client, resolved_settings)

    if not records:
        return {"inserted_count": 0}

    payload = [build_milvus_record(record) for record in records]
    resolved_client.upsert(
        collection_name=resolved_settings.milvus_collection_name,
        data=payload,
    )
    return {"inserted_count": len(payload)}


def build_milvus_record(record: ChunkEmbeddingRecord) -> dict[str, Any]:
    """Convert a chunk embedding record into a Milvus-compatible entity."""
    return {
        "id": record.chunk.chunk_id,
        "vector": record.embedding,
        "document": record.chunk.document,
        "category": record.chunk.category,
        "path": record.chunk.path,
        "title": record.chunk.title,
        "text": record.chunk.text,
    }
