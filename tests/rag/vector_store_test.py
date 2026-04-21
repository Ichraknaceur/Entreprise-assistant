"""Tests for vector store configuration and client creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from pymilvus import DataType

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.rag import vector_store

if TYPE_CHECKING:
    from pathlib import Path

    from enterprise_knowledge_assistant.rag.vector_store import (
        MilvusCollectionClientProtocol,
    )


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


def test_ensure_milvus_collection_creates_expected_schema() -> None:
    """The collection helper should create the configured Milvus schema."""

    class FakeSchema:
        def __init__(self) -> None:
            self.fields: list[dict[str, object]] = []

        def add_field(self, **field_kwargs: object) -> None:
            self.fields.append(field_kwargs)

    class FakeIndexParams:
        def __init__(self) -> None:
            self.indexes: list[dict[str, object]] = []

        def add_index(self, **index_kwargs: object) -> None:
            self.indexes.append(index_kwargs)

    class FakeMilvusClient:
        def __init__(self) -> None:
            self.schema = FakeSchema()
            self.index_params = FakeIndexParams()
            self.created_collection_name: str | None = None

        def has_collection(self, *, collection_name: str) -> bool:
            return False

        def create_schema(
            self,
            *,
            auto_id: bool,
            enable_dynamic_fields: bool,
        ) -> FakeSchema:
            self.schema_config = {
                "auto_id": auto_id,
                "enable_dynamic_fields": enable_dynamic_fields,
            }
            return self.schema

        def prepare_index_params(self) -> FakeIndexParams:
            return self.index_params

        def create_collection(
            self,
            *,
            collection_name: str,
            schema: FakeSchema,
            index_params: FakeIndexParams,
        ) -> None:
            self.created_collection_name = collection_name
            self.created_schema = schema
            self.created_index_params = index_params

    client = FakeMilvusClient()
    settings = Settings(
        milvus_collection_name="knowledge_chunks",
        milvus_embedding_dimension=384,
    )

    vector_store.ensure_milvus_collection(
        cast("MilvusCollectionClientProtocol", client),
        settings,
    )

    assert client.created_collection_name == "knowledge_chunks"
    assert client.schema_config == {
        "auto_id": False,
        "enable_dynamic_fields": False,
    }
    assert client.schema.fields[0] == {
        "field_name": "id",
        "datatype": DataType.VARCHAR,
        "is_primary": True,
        "max_length": 512,
    }
    assert client.schema.fields[1] == {
        "field_name": "vector",
        "datatype": DataType.FLOAT_VECTOR,
        "dim": 384,
    }
    assert client.index_params.indexes == [
        {
            "field_name": "vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
        },
    ]


def test_ensure_milvus_collection_skips_creation_when_existing() -> None:
    """The collection helper should not recreate an existing collection."""

    class FakeMilvusClient:
        def has_collection(self, *, collection_name: str) -> bool:
            return True

        def create_schema(
            self,
            *,
            auto_id: bool,
            enable_dynamic_fields: bool,
        ) -> object:
            msg = "create_schema should not be called"
            raise AssertionError(msg)

    vector_store.ensure_milvus_collection(
        cast("MilvusCollectionClientProtocol", FakeMilvusClient()),
        Settings(),
    )
