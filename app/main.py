"""FastAPI application main entrypoint."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.processing import router as processing_router
from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.graph.neo4j_client import neo4j_client
from app.graph.schema import initialize_schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown hooks."""
    settings = get_settings()
    setup_logging(debug=settings.DEBUG)
    logger.info("Starting up %s in %s mode...", settings.APP_NAME, settings.APP_ENV)

    # Initialize Neo4j connection and schema
    try:
        neo4j_client.connect()
        initialize_schema(neo4j_client)
    except Exception as e:
        logger.error("Failed to connect or initialize Neo4j on startup: %s", e)
        # We don't crash app on startup to allow health checks and container readiness probes

    yield

    # Shutdown
    logger.info("Shutting down %s...", settings.APP_NAME)
    neo4j_client.close()


def create_application() -> FastAPI:
    """Create and configure FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="RegGraph",
        description="Hierarchical and Temporal Regulatory Knowledge Graph Engine",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(processing_router)

    return app


app = create_application()
