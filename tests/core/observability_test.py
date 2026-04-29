"""Tests for Langfuse observability configuration."""

from enterprise_knowledge_assistant.core.config import Settings
from enterprise_knowledge_assistant.core.observability import (
    _NullObservabilityClient,
    get_observability_client,
)


def test_get_observability_client_returns_noop_without_keys() -> None:
    """Observability should no-op when Langfuse credentials are missing."""
    client = get_observability_client(
        Settings(
            langfuse_public_key=None,
            langfuse_secret_key=None,
        ),
    )

    assert isinstance(client, _NullObservabilityClient)
