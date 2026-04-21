"""Health check endpoints."""

from fastapi import APIRouter, HTTPException, status

from enterprise_knowledge_assistant.api.schemas.common import (
    DatabaseHealthResponse,
    HealthResponse,
)
from enterprise_knowledge_assistant.rag.vector_store import check_vector_store_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a simple health response."""
    return HealthResponse(status="ok")


@router.get(
    "/health/database",
    response_model=DatabaseHealthResponse,
    responses={503: {"description": "Vector database is unavailable"}},
)
def database_health_check() -> DatabaseHealthResponse:
    """Return the vector database connectivity status."""
    try:
        health = check_vector_store_health()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector database unavailable: {exc}",
        ) from exc

    return DatabaseHealthResponse(
        status="ok",
        provider=health.provider,
        collection_name=health.collection_name,
        collection_exists=health.collection_exists,
    )
