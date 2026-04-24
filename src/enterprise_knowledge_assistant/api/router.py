"""Aggregate API routes for the application."""

from fastapi import APIRouter

from enterprise_knowledge_assistant.api.routes.health import router as health_router
from enterprise_knowledge_assistant.api.routes.indexing import router as indexing_router
from enterprise_knowledge_assistant.api.routes.query import router as query_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(indexing_router, tags=["indexing"])
api_router.include_router(query_router, tags=["query"])
