"""Tests for the embeddings pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.rag.chunking import KnowledgeChunk
from enterprise_knowledge_assistant.rag.embeddings import build_embeddings

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeEmbeddingModel:
    """Simple fake embedding model used to avoid real model downloads in tests."""

    def __init__(self, vectors: list[list[float]]) -> None:
        """Store vectors to return for a given encode call."""
        self.vectors = vectors
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        """Record encode calls and return the configured vectors."""
        self.calls.append(
            {
                "sentences": list(sentences),
                "normalize_embeddings": normalize_embeddings,
            },
        )
        return self.vectors


def test_build_embeddings_returns_chunk_embedding_records() -> None:
    """Embedding generation should preserve chunk metadata and attach vectors."""
    chunks = [
        KnowledgeChunk(
            chunk_id="hr/remote_work_policy.md::chunk-000",
            document="remote_work_policy.md",
            category="hr",
            path="hr/remote_work_policy.md",
            title="Remote Work Policy",
            text="Employees may work remotely up to three days.",
        ),
        KnowledgeChunk(
            chunk_id="it/vpn_access_guide.md::chunk-000",
            document="vpn_access_guide.md",
            category="it",
            path="it/vpn_access_guide.md",
            title="VPN Access Guide",
            text="VPN access requires MFA and an approved device.",
        ),
    ]
    model = FakeEmbeddingModel(
        vectors=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    records = build_embeddings(chunks, model=model)

    assert len(records) == 2
    assert records[0].chunk.chunk_id == "hr/remote_work_policy.md::chunk-000"
    assert records[0].embedding == [0.1, 0.2, 0.3]
    assert records[1].chunk.chunk_id == "it/vpn_access_guide.md::chunk-000"
    assert records[1].embedding == [0.4, 0.5, 0.6]
    assert model.calls == [
        {
            "sentences": [
                "Employees may work remotely up to three days.",
                "VPN access requires MFA and an approved device.",
            ],
            "normalize_embeddings": True,
        },
    ]


def test_build_embeddings_returns_empty_list_for_empty_chunks() -> None:
    """Embedding generation should no-op cleanly on empty input."""
    model = FakeEmbeddingModel(vectors=[])

    records = build_embeddings([], model=model)

    assert records == []
    assert model.calls == []
