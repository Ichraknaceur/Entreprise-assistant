"""Service layer for local knowledge-base indexing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.rag.chunking import chunk_documents
from enterprise_knowledge_assistant.rag.embeddings import build_embeddings
from enterprise_knowledge_assistant.rag.ingestion import ingest_documents
from enterprise_knowledge_assistant.rag.loaders import load_markdown_documents

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.config import Settings
    from enterprise_knowledge_assistant.rag.embeddings import EmbeddingModelProtocol
    from enterprise_knowledge_assistant.rag.ingestion import (
        MilvusIngestionClientProtocol,
    )


@dataclass(slots=True, frozen=True)
class IndexingResult:
    """Summary of an indexing run."""

    documents_count: int
    chunks_count: int
    inserted_count: int


class IndexingService:
    """Coordinate end-to-end indexing from markdown files into Milvus."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedding_model: EmbeddingModelProtocol | None = None,
        milvus_client: MilvusIngestionClientProtocol | None = None,
    ) -> None:
        """Initialize the indexing service with optional test doubles."""
        self._settings = settings or get_settings()
        self._embedding_model = embedding_model
        self._milvus_client = milvus_client

    def index_documents(self) -> IndexingResult:
        """Load, chunk, embed, and ingest the local knowledge base."""
        documents = load_markdown_documents(self._settings.data_dir)
        chunks = chunk_documents(documents)
        records = build_embeddings(
            chunks,
            settings=self._settings,
            model=self._embedding_model,
        )
        ingestion_result = ingest_documents(
            records,
            settings=self._settings,
            client=cast("MilvusIngestionClientProtocol | None", self._milvus_client),
        )
        return IndexingResult(
            documents_count=len(documents),
            chunks_count=len(chunks),
            inserted_count=ingestion_result["inserted_count"],
        )
