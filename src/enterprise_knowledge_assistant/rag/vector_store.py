"""Vector store integration points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pymilvus import MilvusClient

from enterprise_knowledge_assistant.core.config import get_settings

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.core.config import Settings


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


def _resolve_milvus_uri(settings: Settings) -> str:
    """Resolve the configured Milvus URI, defaulting to a local Milvus Lite file."""
    if settings.milvus_uri:
        return settings.milvus_uri
    return str(settings.vector_store_dir / "enterprise_knowledge_assistant.db")


def _is_local_milvus_uri(uri: str) -> bool:
    """Return whether the URI should be treated as a local Milvus Lite database."""
    parsed_uri = urlparse(uri)
    return parsed_uri.scheme == ""
