"""Tests for the query service orchestration."""

from typing import TYPE_CHECKING, cast

from enterprise_knowledge_assistant.api.schemas.query import QueryRequest
from enterprise_knowledge_assistant.rag.retriever import RetrievedChunk
from enterprise_knowledge_assistant.services.query_service import QueryService

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


class FakeObservation:
    """Simple fake observation/span for tracing tests."""

    def __init__(self, name: str, as_type: str) -> None:
        """Initialize call storage for the observation."""
        self.name = name
        self.as_type = as_type
        self.updates: list[dict[str, object]] = []

    def update(self, **kwargs: object) -> None:
        """Record observation updates."""
        self.updates.append(kwargs)


class FakeObservabilityClient:
    """Simple fake observability client for query service tests."""

    def __init__(self) -> None:
        """Initialize created observation storage."""
        self.observations: list[FakeObservation] = []
        self.flush_calls = 0

    class _ContextManager:
        """Simple context manager wrapper around a fake observation."""

        def __init__(self, observation: FakeObservation) -> None:
            """Store the wrapped observation."""
            self._observation = observation

        def __enter__(self) -> FakeObservation:
            """Enter the context."""
            return self._observation

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> bool:
            """Leave the context without suppressing exceptions."""
            del exc_type, exc, tb
            return False

    def start_as_current_observation(  # noqa: PLR0913
        self,
        *,
        name: str,
        as_type: str = "span",
        input: object | None = None,  # noqa: A002
        output: object | None = None,
        metadata: object | None = None,
        model: str | None = None,
    ) -> _ContextManager:
        """Create and return a fake observation context manager."""
        observation = FakeObservation(name=name, as_type=as_type)
        observation.update(
            input=input,
            output=output,
            metadata=metadata,
            model=model,
        )
        self.observations.append(observation)
        return self._ContextManager(observation)

    def flush(self) -> None:
        """No-op flush for test compatibility."""
        self.flush_calls += 1


class FakeGenerator:
    """Simple fake generator for query service tests."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.calls: list[dict[str, object]] = []

    @property
    def provider_name(self) -> str:
        """Return the fake provider name."""
        return "mock"

    @property
    def model_name(self) -> str:
        """Return the fake model name."""
        return "mock-generator"

    def generate(self, question: str, contexts: list[str]) -> str:
        """Record generation inputs and return a deterministic answer."""
        self.calls.append({"question": question, "contexts": contexts})
        return "Mock answer grounded in retrieved contexts."


def test_query_service_returns_retrieved_sources() -> None:
    """Query service should combine retriever output with generator output."""
    generator = FakeGenerator()
    observability_client = FakeObservabilityClient()
    selected_providers: list[str | None] = []

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

    def fake_generator_factory(provider: str | None) -> BaseGenerator:
        selected_providers.append(provider)
        return cast("BaseGenerator", generator)

    service = QueryService(
        default_provider="mock",
        generator_factory=fake_generator_factory,
        retriever=fake_retriever,
        observability_client=observability_client,
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
    assert selected_providers == ["mock"]
    assert [obs.name for obs in observability_client.observations] == [
        "query-request",
        "retrieve-context",
        "generate-answer",
    ]
    assert observability_client.flush_calls == 1


def test_query_service_uses_provider_override() -> None:
    """Query service should allow the request to override the provider."""
    generator = FakeGenerator()
    observability_client = FakeObservabilityClient()
    selected_providers: list[str | None] = []

    def fake_retriever(question: str) -> list[RetrievedChunk]:
        del question
        return [
            RetrievedChunk(
                chunk_id="it/vpn_access_guide.md::chunk-000",
                document="vpn_access_guide.md",
                category="it",
                path="it/vpn_access_guide.md",
                title="VPN Access Guide",
                text="VPN access requires MFA and an approved device.",
                score=0.91,
            ),
        ]

    def fake_generator_factory(provider: str | None) -> BaseGenerator:
        selected_providers.append(provider)
        return cast("BaseGenerator", generator)

    service = QueryService(
        default_provider="mock",
        generator_factory=fake_generator_factory,
        retriever=fake_retriever,
        observability_client=observability_client,
    )

    result = service.query(
        QueryRequest(
            question="How do I access the VPN?",
            provider="openai",
        ),
    )

    assert result.answer == "Mock answer grounded in retrieved contexts."
    assert selected_providers == ["openai"]
    assert observability_client.flush_calls == 1


def test_query_service_returns_insufficient_context_answer() -> None:
    """Query service should short-circuit when retrieval finds no context."""
    generator = FakeGenerator()
    observability_client = FakeObservabilityClient()

    def fake_retriever(question: str) -> list[RetrievedChunk]:
        del question
        return []

    def fake_generator_factory(provider: str | None) -> BaseGenerator:
        msg = "generator should not be called when no context is retrieved"
        raise AssertionError(msg)

    service = QueryService(
        default_provider="mock",
        generator_factory=fake_generator_factory,
        retriever=fake_retriever,
        observability_client=observability_client,
    )

    result = service.query(
        QueryRequest(question="What is the vendor review exception process?"),
    )

    assert result.answer == (
        "I do not have enough information in the knowledge base to answer "
        "that question."
    )
    assert result.sources == []
    assert generator.calls == []
    assert [obs.name for obs in observability_client.observations] == [
        "query-request",
        "retrieve-context",
    ]
    assert observability_client.flush_calls == 1
