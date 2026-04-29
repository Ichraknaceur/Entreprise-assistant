"""Shared dependency providers for API routes."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.rag.generator.factory import get_generator
from enterprise_knowledge_assistant.rag.retriever import retrieve_context
from enterprise_knowledge_assistant.services.indexing_service import IndexingService
from enterprise_knowledge_assistant.services.query_service import QueryService

if TYPE_CHECKING:
    from collections.abc import Callable

    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


@lru_cache
def _get_generator(provider: str | None = None) -> BaseGenerator:
    """Build the configured answer generator."""
    settings = get_settings()
    return get_generator(settings, provider=provider)


def _get_generator_factory() -> Callable[[str | None], BaseGenerator]:
    """Return a cached generator factory."""
    return _get_generator


@lru_cache
def get_indexing_service() -> IndexingService:
    """Build the indexing service."""
    settings = get_settings()
    return IndexingService(settings=settings)


@lru_cache
def get_query_service() -> QueryService:
    """Build the query service."""
    settings = get_settings()
    return QueryService(
        default_provider=settings.llm_provider,
        generator_factory=_get_generator_factory(),
        retriever=retrieve_context,
    )
