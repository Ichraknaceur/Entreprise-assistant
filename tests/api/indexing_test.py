"""Tests for the indexing API route."""

from fastapi.testclient import TestClient

from enterprise_knowledge_assistant.core.dependencies import get_indexing_service
from enterprise_knowledge_assistant.main import app
from enterprise_knowledge_assistant.services.indexing_service import IndexingResult

client = TestClient(app)


class FakeIndexingService:
    """Simple fake indexing service used for API tests."""

    def index_documents(self) -> IndexingResult:
        """Return a stable indexing summary."""
        return IndexingResult(
            documents_count=3,
            chunks_count=7,
            inserted_count=7,
        )


def test_indexing_endpoint_returns_summary() -> None:
    """The indexing endpoint should return the indexing run summary."""
    app.dependency_overrides[get_indexing_service] = FakeIndexingService

    try:
        response = client.post("/admin/index")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "documents_count": 3,
        "chunks_count": 7,
        "inserted_count": 7,
    }
