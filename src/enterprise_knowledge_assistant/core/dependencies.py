"""Shared dependency providers for API routes."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.rag.generator.factory import get_generator
from enterprise_knowledge_assistant.services.query_service import QueryService

if TYPE_CHECKING:
    from enterprise_knowledge_assistant.rag.generator.base import BaseGenerator


@lru_cache
def _get_generator() -> BaseGenerator:
    """Build the configured answer generator."""
    settings = get_settings()
    return get_generator(settings)


@lru_cache
def get_query_service() -> QueryService:
    """Build the query service."""
    return QueryService(generator=_get_generator())
