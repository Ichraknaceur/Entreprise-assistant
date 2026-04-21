"""Vector store integration points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from pymilvus import DataType, MilvusClient

from enterprise_knowledge_assistant.core.config import get_settings

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.config import Settings


class MilvusSchemaProtocol(Protocol):
    """Protocol for the Milvus schema builder used by collection creation."""

    def add_field(self, **field_kwargs: object) -> None:
        """Add a field definition to the schema."""


class MilvusIndexParamsProtocol(Protocol):
    """Protocol for the Milvus index params builder."""

    def add_index(self, **index_kwargs: object) -> None:
        """Add an index definition."""


class MilvusCollectionClientProtocol(Protocol):
    """Protocol for the subset of Milvus client APIs used by this project."""

    def list_collections(self) -> list[str]:
        """Return the available collection names."""

    def has_collection(self, *, collection_name: str) -> bool:
        """Return whether a collection exists."""

    def create_schema(
        self,
        *,
        auto_id: bool,
        enable_dynamic_fields: bool,
    ) -> MilvusSchemaProtocol:
        """Create a schema builder."""

    def prepare_index_params(self) -> MilvusIndexParamsProtocol:
        """Create an index parameter builder."""

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: MilvusSchemaProtocol,
        index_params: MilvusIndexParamsProtocol,
    ) -> None:
        """Create a collection."""


@dataclass(slots=True, frozen=True)
class VectorStoreHealth:
    """Resolved health information for the configured vector store."""

    provider: str
    collection_name: str
    collection_exists: bool


def get_vector_store(settings: Settings | None = None) -> MilvusClient:
    """Return the configured vector store client."""
    resolved_settings = settings or get_settings()
    if resolved_settings.vector_db_provider != "milvus":
        msg = (
            "Unsupported vector database provider: "
            f"{resolved_settings.vector_db_provider}"
        )
        raise ValueError(msg)

    return create_milvus_client(
        uri=_resolve_milvus_uri(resolved_settings),
        token=resolved_settings.milvus_token,
    )


def check_vector_store_health(
    settings: Settings | None = None,
) -> VectorStoreHealth:
    """Check whether the configured vector store is reachable."""
    resolved_settings = settings or get_settings()
    client = get_vector_store(resolved_settings)
    collection_names = client.list_collections()
    return VectorStoreHealth(
        provider=resolved_settings.vector_db_provider,
        collection_name=resolved_settings.milvus_collection_name,
        collection_exists=resolved_settings.milvus_collection_name in collection_names,
    )


def create_milvus_client(*, uri: str, token: str | None = None) -> MilvusClient:
    """Create a Milvus client for either Milvus Lite or a remote Milvus server."""
    if _is_local_milvus_uri(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)

    if token:
        return MilvusClient(uri=uri, token=token)
    return MilvusClient(uri=uri)


def ensure_milvus_collection(
    client: MilvusCollectionClientProtocol,
    settings: Settings | None = None,
) -> None:
    """Create the configured Milvus collection if it does not yet exist."""
    resolved_settings = settings or get_settings()
    collection_name = resolved_settings.milvus_collection_name
    if client.has_collection(collection_name=collection_name):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_fields=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=512,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=resolved_settings.milvus_embedding_dimension,
    )
    schema.add_field(
        field_name="document",
        datatype=DataType.VARCHAR,
        max_length=255,
    )
    schema.add_field(
        field_name="category",
        datatype=DataType.VARCHAR,
        max_length=100,
    )
    schema.add_field(
        field_name="path",
        datatype=DataType.VARCHAR,
        max_length=512,
    )
    schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=255,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=8192,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def _resolve_milvus_uri(settings: Settings) -> str:
    """Resolve the configured Milvus URI, defaulting to a local Milvus Lite file."""
    if settings.milvus_uri:
        return settings.milvus_uri
    return str(settings.vector_store_dir / "enterprise_knowledge_assistant.db")


def _is_local_milvus_uri(uri: str) -> bool:
    """Return whether the URI should be treated as a local Milvus Lite database."""
    parsed_uri = urlparse(uri)
    return parsed_uri.scheme == ""
