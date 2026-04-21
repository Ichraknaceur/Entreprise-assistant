"""Tests for health and skeleton query routes."""

from fastapi.testclient import TestClient

from enterprise_knowledge_assistant.main import app

client = TestClient(app)


def test_health_check() -> None:
    """The health endpoint should return an OK response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
