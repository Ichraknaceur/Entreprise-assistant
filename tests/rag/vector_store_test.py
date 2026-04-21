"""Tests for vector store configuration and client creation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.rag import vector_store

if TYPE_CHECKING:
    from pathlib import Path


def test_get_vector_store_uses_local_milvus_lite_file(tmp_path: Path) -> None:
    """The vector store should default to a local Milvus Lite database file."""
    captured_kwargs: dict[str, str] = {}

    class FakeMilvusClient:
        """Simple fake used to capture initialization arguments."""

        def __init__(self, *, uri: str, token: str | None = None) -> None:
            captured_kwargs["uri"] = uri
            if token is not None:
                captured_kwargs["token"] = token

    settings = Settings(
        vector_db_provider="milvus",
        vector_store_dir=tmp_path / "milvus",
    )

    with patch(
        "enterprise_knowledge_assistant.rag.vector_store.MilvusClient",
        FakeMilvusClient,
    ):
        client = vector_store.get_vector_store(settings)

    assert isinstance(client, FakeMilvusClient)
    assert captured_kwargs["uri"].endswith("enterprise_knowledge_assistant.db")
    assert (tmp_path / "milvus").exists()


def test_get_vector_store_passes_remote_uri_and_token() -> None:
    """The vector store should forward remote Milvus credentials unchanged."""
    captured_kwargs: dict[str, str] = {}
    auth_token = "test-token"  # noqa: S105

    class FakeMilvusClient:
        """Simple fake used to capture initialization arguments."""

        def __init__(self, *, uri: str, token: str | None = None) -> None:
            captured_kwargs["uri"] = uri
            if token is not None:
                captured_kwargs["token"] = token

    settings = Settings(
        vector_db_provider="milvus",
        milvus_uri="http://localhost:19530",
        milvus_token=auth_token,
    )

    with patch(
        "enterprise_knowledge_assistant.rag.vector_store.MilvusClient",
        FakeMilvusClient,
    ):
        client = vector_store.get_vector_store(settings)

    assert isinstance(client, FakeMilvusClient)
    assert captured_kwargs == {
        "uri": "http://localhost:19530",
        "token": auth_token,
    }


def test_get_vector_store_rejects_unsupported_provider() -> None:
    """The vector store should fail fast on unknown providers."""
    settings = Settings(vector_db_provider="pinecone")

    with pytest.raises(ValueError, match="Unsupported vector database provider"):
        vector_store.get_vector_store(settings)
