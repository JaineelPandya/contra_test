"""Health check endpoint."""
from typing import Dict
from fastapi import APIRouter
from app.graph.neo4j_client import neo4j_client

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Check API server and Neo4j database health status."""
    is_neo4j_alive = neo4j_client.is_connected()
    return {
        "status": "healthy" if is_neo4j_alive else "degraded",
        "neo4j": "connected" if is_neo4j_alive else "disconnected",
    }
