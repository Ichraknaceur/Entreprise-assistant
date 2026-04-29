"""Embedding model integration points."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

from sentence_transformers import SentenceTransformer

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.rag.ingestion import ChunkEmbeddingRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.rag.chunking import KnowledgeChunk


class EmbeddingModelProtocol(Protocol):
    """Protocol for the embedding model used by the pipeline."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> object:
        """Encode input texts into embeddings."""


@lru_cache
def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Return the configured sentence-transformers model."""
    resolved_model_name = model_name or get_settings().embedding_model_name
    return SentenceTransformer(resolved_model_name)


def build_embeddings(
    chunks: Sequence[KnowledgeChunk],
    *,
    settings: Settings | None = None,
    model: EmbeddingModelProtocol | None = None,
) -> list[ChunkEmbeddingRecord]:
    """Create embeddings for document chunks."""
    if not chunks:
        return []

    resolved_model = model or get_embedding_model(
        (settings or get_settings()).embedding_model_name,
    )
    raw_embeddings = resolved_model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
    )
    embeddings = cast("Sequence[Sequence[float]]", raw_embeddings)

    return [
        ChunkEmbeddingRecord(
            chunk=chunk,
            embedding=[float(value) for value in embedding],
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def build_query_embedding(
    question: str,
    *,
    settings: Settings | None = None,
    model: EmbeddingModelProtocol | None = None,
) -> list[float]:
    """Create a normalized embedding vector for a user question."""
    resolved_model = model or get_embedding_model(
        (settings or get_settings()).embedding_model_name,
    )
    raw_embedding = resolved_model.encode(
        [question],
        normalize_embeddings=True,
    )
    embedding = cast("Sequence[Sequence[float]]", raw_embedding)[0]
    return [float(value) for value in embedding]
