"""Administrative routes for indexing the local knowledge base."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from enterprise_knowledge_assistant.api.schemas.indexing import IndexResponse
from enterprise_knowledge_assistant.core.dependencies import get_indexing_service
from enterprise_knowledge_assistant.services.indexing_service import (
    IndexingService,  # noqa: TC001
)

router = APIRouter()


@router.post("/admin/index", response_model=IndexResponse)
def index_knowledge_base(
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> IndexResponse:
    """Trigger a local knowledge-base indexing run."""
    result = indexing_service.index_documents()
    return IndexResponse(
        status="ok",
        documents_count=result.documents_count,
        chunks_count=result.chunks_count,
        inserted_count=result.inserted_count,
    )
