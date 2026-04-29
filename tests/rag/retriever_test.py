"""Tests for the Milvus-backed retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.rag.retriever import retrieve_context

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeEmbeddingModel:
    """Simple fake embedding model for retrieval tests."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        """Return one stable embedding for the incoming query."""
        self.calls.append(
            {
                "sentences": list(sentences),
                "normalize_embeddings": normalize_embeddings,
            },
        )
        return [[0.1, 0.2, 0.3]]


class FakeMilvusClient:
    """Simple fake Milvus search client."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        collection_name: str,
        *,
        data: list[list[float]],
        limit: int,
        output_fields: list[str],
        search_params: dict[str, str],
    ) -> list[list[dict[str, object]]]:
        """Record the search call and return deterministic results."""
        self.calls.append(
            {
                "collection_name": collection_name,
                "data": data,
                "limit": limit,
                "output_fields": output_fields,
                "search_params": search_params,
            },
        )
        return [
            [
                {
                    "id": "hr/remote_work_policy.md::chunk-000",
                    "distance": 0.93,
                    "entity": {
                        "id": "hr/remote_work_policy.md::chunk-000",
                        "document": "remote_work_policy.md",
                        "category": "hr",
                        "path": "hr/remote_work_policy.md",
                        "title": "Remote Work Policy",
                        "text": "Employees may work remotely up to three days.",
                    },
                },
            ],
        ]


def test_retrieve_context_returns_normalized_chunks() -> None:
    """Retriever should embed the query and map Milvus hits to app models."""
    settings = Settings(
        milvus_collection_name="knowledge_chunks",
        milvus_embedding_dimension=3,
    )
    model = FakeEmbeddingModel()
    client = FakeMilvusClient()

    results = retrieve_context(
        "What is the remote work policy?",
        settings=settings,
        model=model,
        client=client,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "hr/remote_work_policy.md::chunk-000"
    assert results[0].document == "remote_work_policy.md"
    assert results[0].text == "Employees may work remotely up to three days."
    assert results[0].score == 0.93
    assert model.calls == [
        {
            "sentences": ["What is the remote work policy?"],
            "normalize_embeddings": True,
        },
    ]
    assert client.calls == [
        {
            "collection_name": "knowledge_chunks",
            "data": [[0.1, 0.2, 0.3]],
            "limit": 3,
            "output_fields": ["document", "category", "path", "title", "text"],
            "search_params": {"metric_type": "COSINE"},
        },
    ]


def test_retrieve_context_returns_empty_list_when_no_hits() -> None:
    """Retriever should return an empty list when Milvus finds no matches."""

    class EmptyMilvusClient:
        """Simple fake returning no search hits."""

        def search(
            self,
            collection_name: str,
            *,
            data: list[list[float]],
            limit: int,
            output_fields: list[str],
            search_params: dict[str, str],
        ) -> list[list[dict[str, object]]]:
            """Return an empty search result set."""
            del collection_name, data, limit, output_fields, search_params
            return []

    results = retrieve_context(
        "How do I request software access?",
        model=FakeEmbeddingModel(),
        client=EmptyMilvusClient(),
    )

    assert results == []
