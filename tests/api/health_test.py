"""Tests for health and skeleton query routes."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from enterprise_knowledge_assistant.main import app
from enterprise_knowledge_assistant.rag.vector_store import VectorStoreHealth

client = TestClient(app)


def test_health_check() -> None:
    """The health endpoint should return an OK response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_database_health_check_returns_ok() -> None:
    """The database health endpoint should return vector store status."""
    with patch(
        "enterprise_knowledge_assistant.api.routes.health.check_vector_store_health",
        return_value=VectorStoreHealth(
            provider="milvus",
            collection_name="knowledge_chunks",
            collection_exists=False,
        ),
    ):
        response = client.get("/health/database")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "provider": "milvus",
        "collection_name": "knowledge_chunks",
        "collection_exists": False,
    }


def test_database_health_check_returns_503_when_unavailable() -> None:
    """The database health endpoint should surface vector store failures."""
    with patch(
        "enterprise_knowledge_assistant.api.routes.health.check_vector_store_health",
        side_effect=RuntimeError("connection refused"),
    ):
        response = client.get("/health/database")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Vector database unavailable: connection refused",
    }


def test_query_returns_placeholder_payload() -> None:
    """The query endpoint should return the expected response shape."""
    response = client.post(
        "/query",
        json={"question": "What is the remote work policy?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "sources" in payload
