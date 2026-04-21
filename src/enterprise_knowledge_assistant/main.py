"""Application entrypoint for the Enterprise Knowledge Assistant."""

import uvicorn
from fastapi import FastAPI

from enterprise_knowledge_assistant.api.router import api_router
from enterprise_knowledge_assistant.core.config import get_settings
from enterprise_knowledge_assistant.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()


def run() -> None:
    """Run the application with uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "enterprise_knowledge_assistant.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
