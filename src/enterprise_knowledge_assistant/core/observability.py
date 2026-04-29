"""Observability integration points."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

from langfuse import Langfuse

from enterprise_knowledge_assistant.core.config import get_settings

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from enterprise_knowledge_assistant.core.config import Settings


class ObservationProtocol(Protocol):
    """Protocol for a trace observation/span."""

    def update(self, **kwargs: object) -> None:
        """Update observation metadata, input, or output."""


class ObservabilityClientProtocol(Protocol):
    """Protocol for the observability client used by the app."""

    def start_as_current_observation(  # noqa: PLR0913
        self,
        *,
        name: str,
        as_type: str = "span",
        input: object | None = None,  # noqa: A002
        output: object | None = None,
        metadata: object | None = None,
        model: str | None = None,
    ) -> AbstractContextManager[ObservationProtocol]:
        """Start an observation/span as the current context."""

    def flush(self) -> None:
        """Flush pending telemetry events."""


class _NullObservation:
    """No-op observation returned when Langfuse is not configured."""

    def update(self, **kwargs: object) -> None:
        """Ignore updates."""


class _NullObservabilityClient:
    """No-op client used when Langfuse is not configured."""

    def start_as_current_observation(  # noqa: PLR0913
        self,
        *,
        name: str,
        as_type: str = "span",
        input: object | None = None,  # noqa: A002
        output: object | None = None,
        metadata: object | None = None,
        model: str | None = None,
    ) -> AbstractContextManager[ObservationProtocol]:
        """Return a no-op context manager."""
        del name, as_type, input, output, metadata, model
        return nullcontext(_NullObservation())

    def flush(self) -> None:
        """No-op flush."""


def get_observability_client(
    settings: Settings | None = None,
) -> ObservabilityClientProtocol:
    """Return the configured observability client or a no-op fallback."""
    resolved_settings = settings or get_settings()
    if not (
        resolved_settings.langfuse_public_key and resolved_settings.langfuse_secret_key
    ):
        return _NullObservabilityClient()

    return cast(
        "ObservabilityClientProtocol",
        _build_langfuse_client(
            public_key=resolved_settings.langfuse_public_key,
            secret_key=resolved_settings.langfuse_secret_key,
            base_url=resolved_settings.langfuse_base_url,
            environment=resolved_settings.langfuse_environment,
            release=resolved_settings.app_version,
        ),
    )


@lru_cache
def _build_langfuse_client(
    *,
    public_key: str,
    secret_key: str,
    base_url: str,
    environment: str,
    release: str,
) -> Langfuse:
    """Build a cached Langfuse client instance."""
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        release=release,
    )
