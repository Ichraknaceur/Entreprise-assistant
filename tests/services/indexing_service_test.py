"""Tests for the end-to-end indexing service."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.services.indexing_service import IndexingService

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from enterprise_knowledge_assistant.rag.ingestion import (
        MilvusIngestionClientProtocol,
    )


class FakeEmbeddingModel:
    """Small fake embedding model used to avoid real model loading in tests."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        """Return one stable vector per input sentence."""
        assert normalize_embeddings is True
        return [[float(index), 0.1, 0.2] for index, _ in enumerate(sentences, start=1)]


class FakeSchema:
    """Schema fake used by the Milvus test double."""

    def add_field(self, **field_kwargs: object) -> None:
        """Accept schema field definitions without storing them."""


class FakeIndexParams:
    """Index parameter fake used by the Milvus test double."""

    def add_index(self, **index_kwargs: object) -> None:
        """Accept index definitions without storing them."""


class FakeMilvusClient:
    """Simple fake used to capture collection creation and upserts."""

    def __init__(self) -> None:
        """Initialize fake call storage."""
        self.upsert_calls: list[tuple[str, list[dict[str, object]]]] = []
        self.collections: set[str] = set()

    def has_collection(self, *, collection_name: str) -> bool:
        """Return whether the collection has already been created."""
        return collection_name in self.collections

    def create_schema(
        self,
        *,
        auto_id: bool,
        enable_dynamic_fields: bool,
    ) -> FakeSchema:
        """Return a fake schema builder."""
        return FakeSchema()

    def prepare_index_params(self) -> FakeIndexParams:
        """Return a fake index parameter builder."""
        return FakeIndexParams()

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: object,
        index_params: object,
    ) -> None:
        """Record creation of the configured collection."""
        del schema, index_params
        self.collections.add(collection_name)

    def upsert(self, *, collection_name: str, data: list[dict[str, object]]) -> None:
        """Record upsert calls."""
        self.upsert_calls.append((collection_name, data))


def test_index_documents_runs_end_to_end(tmp_path: Path) -> None:
    """Indexing should load local docs and persist embedded chunks."""
    data_dir = tmp_path / "sample_docs"
    hr_dir = data_dir / "hr"
    it_dir = data_dir / "it"
    hr_dir.mkdir(parents=True)
    it_dir.mkdir(parents=True)
    (hr_dir / "remote_work_policy.md").write_text(
        "# Remote Work Policy\n\nEmployees may work remotely up to three days.",
        encoding="utf-8",
    )
    (it_dir / "vpn_access_guide.md").write_text(
        "# VPN Access Guide\n\nVPN access requires MFA and an approved device.",
        encoding="utf-8",
    )

    settings = Settings(
        data_dir=data_dir,
        milvus_collection_name="knowledge_chunks",
        milvus_embedding_dimension=3,
    )
    service = IndexingService(
        settings=settings,
        embedding_model=FakeEmbeddingModel(),
        milvus_client=cast("MilvusIngestionClientProtocol", FakeMilvusClient()),
    )

    result = service.index_documents()

    assert result.documents_count == 2
    assert result.chunks_count == 2
    assert result.inserted_count == 2
