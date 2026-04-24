"""Schemas for indexing requests and responses."""

from enterprise_knowledge_assistant.api.schemas.common import APIModel


class IndexResponse(APIModel):
    """Response payload for a knowledge-base indexing run."""

    status: str
    documents_count: int
    chunks_count: int
    inserted_count: int
