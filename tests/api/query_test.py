"""Tests for the query API route."""

from fastapi.testclient import TestClient

from enterprise_knowledge_assistant.api.schemas.query import QueryResponse, SourceItem
from enterprise_knowledge_assistant.core.dependencies import get_query_service
from enterprise_knowledge_assistant.main import app

client = TestClient(app)


class FakeQueryService:
    """Simple fake query service used for API tests."""

    def query(self, request: object) -> QueryResponse:
        """Return a deterministic query response."""
        del request
        return QueryResponse(
            answer="Mock answer grounded in retrieved contexts.",
            sources=[
                SourceItem(
                    document="remote_work_policy.md",
                    snippet="Employees may work remotely up to three days.",
                ),
            ],
        )


def test_query_returns_retrieved_sources() -> None:
    """The query endpoint should return a grounded answer payload."""
    app.dependency_overrides[get_query_service] = FakeQueryService

    try:
        response = client.post(
            "/query",
            json={"question": "What is the remote work policy?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Mock answer grounded in retrieved contexts.",
        "sources": [
            {
                "document": "remote_work_policy.md",
                "snippet": "Employees may work remotely up to three days.",
            },
        ],
    }


def test_query_accepts_provider_override() -> None:
    """The query endpoint should accept a supported provider override."""
    app.dependency_overrides[get_query_service] = FakeQueryService

    try:
        response = client.post(
            "/query",
            json={
                "question": "What is the remote work policy?",
                "provider": "openai",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


def test_query_returns_503_for_generator_configuration_errors() -> None:
    """The query endpoint should surface generator configuration failures."""

    class FailingQueryService:
        """Fake query service raising a configuration error."""

        def query(self, request: object) -> QueryResponse:
            """Raise the expected configuration error."""
            del request
            msg = "OpenAI API key is not configured."
            raise ValueError(msg)

    app.dependency_overrides[get_query_service] = FailingQueryService

    try:
        response = client.post(
            "/query",
            json={"question": "What is the remote work policy?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "OpenAI API key is not configured."}
