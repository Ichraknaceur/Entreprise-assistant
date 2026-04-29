"""Tests for the query service orchestration."""

from typing import TYPE_CHECKING, cast

from enterprise_knowledge_assistant.api.schemas.query import QueryRequest
from enterprise_knowledge_assistant.rag.retriever import RetrievedChunk
from enterprise_knowledge_assistant.services.query_service import QueryService

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


class FakeGenerator:
    """Simple fake generator for query service tests."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.calls: list[dict[str, object]] = []

    def generate(self, question: str, contexts: list[str]) -> str:
        """Record generation inputs and return a deterministic answer."""
        self.calls.append({"question": question, "contexts": contexts})
        return "Mock answer grounded in retrieved contexts."


def test_query_service_returns_retrieved_sources() -> None:
    """Query service should combine retriever output with generator output."""
    generator = FakeGenerator()

    def fake_retriever(question: str) -> list[RetrievedChunk]:
        assert question == "What is the remote work policy?"
        return [
            RetrievedChunk(
                chunk_id="hr/remote_work_policy.md::chunk-000",
                document="remote_work_policy.md",
                category="hr",
                path="hr/remote_work_policy.md",
                title="Remote Work Policy",
                text="Employees may work remotely up to three days.",
                score=0.93,
            ),
        ]

    service = QueryService(
        generator=cast("BaseGenerator", generator),
        retriever=fake_retriever,
    )

    result = service.query(
        QueryRequest(question="What is the remote work policy?"),
    )

    assert result.answer == "Mock answer grounded in retrieved contexts."
    assert [source.model_dump() for source in result.sources] == [
        {
            "document": "remote_work_policy.md",
            "snippet": "Employees may work remotely up to three days.",
        },
    ]
    assert generator.calls == [
        {
            "question": "What is the remote work policy?",
            "contexts": ["Employees may work remotely up to three days."],
        },
    ]
