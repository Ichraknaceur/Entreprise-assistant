"""Question-answering API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from enterprise_knowledge_assistant.api.schemas.query import (  # noqa: TC001
    QueryRequest,
    QueryResponse,
)
from enterprise_knowledge_assistant.core.dependencies import get_query_service
from enterprise_knowledge_assistant.services.query_service import (
    QueryService,  # noqa: TC001
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    query_service: Annotated[QueryService, Depends(get_query_service)],
) -> QueryResponse:
    """Return an answer backed by retrieved sources."""
    try:
        return query_service.query(request)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
