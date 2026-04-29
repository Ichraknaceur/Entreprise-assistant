"""Schemas for query requests and responses."""

from typing import Literal

from pydantic import Field

from enterprise_knowledge_assistant.api.schemas.common import APIModel


class QueryRequest(APIModel):
    """Incoming user query."""

    question: str = Field(min_length=3, max_length=500)
    provider: Literal["mock", "openai"] | None = None


class SourceItem(APIModel):
    """A source document citation returned with an answer."""

    document: str
    snippet: str


class QueryResponse(APIModel):
    """Response payload for a knowledge query."""

    answer: str
    sources: list[SourceItem]
