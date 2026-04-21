"""Shared API schema models."""

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    """Base schema with consistent Pydantic configuration."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HealthResponse(APIModel):
    """Health check response payload."""

    status: str


class DatabaseHealthResponse(APIModel):
    """Vector database health response payload."""

    status: str
    provider: str
    collection_name: str
    collection_exists: bool
