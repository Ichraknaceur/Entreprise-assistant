"""Health check endpoints."""

from fastapi import APIRouter

from enterprise_knowledge_assistant.api.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a simple health response."""
    return HealthResponse(status="ok")
