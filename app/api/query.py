"""FastAPI Router for Regulatory Q&A, Graph Queries, and Visualization."""
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from app.graph.neo4j_client import neo4j_client
from app.graph.repository import GraphRepository
from app.llm.factory import get_llm_provider
from app.models.query import (
    QueryType,
    RegulatoryQueryRequest,
    RegulatoryQueryResponse,
)
from app.services.query_service import RegulatoryQueryService

router = APIRouter(prefix="/api/query", tags=["Regulatory Query Engine"])


def get_query_service() -> RegulatoryQueryService:
    """Dependency injector for RegulatoryQueryService."""
    repo = GraphRepository(neo4j_client)
    llm = get_llm_provider()
    return RegulatoryQueryService(repository=repo, llm_provider=llm)


@router.post(
    "",
    response_model=RegulatoryQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute regulatory query with dual-path context and subgraph output",
    description=(
        "Executes a natural language regulatory query. Analyzes intent, plans graph traversal across "
        "5 archetypes (Current, Historical, Evolution, Amendment, Master→Latest), generates & validates Cypher, "
        "and returns current active state, chronological change history, synthesized answer, and visual subgraph."
    ),
)
async def execute_regulatory_query(
    request: RegulatoryQueryRequest,
    service: RegulatoryQueryService = Depends(get_query_service),
) -> RegulatoryQueryResponse:
    """Execute end-to-end regulatory query."""
    try:
        return await service.execute_query(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post(
    "/plan",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Preview query plan and generated Cypher query",
    description="Analyzes the query and generates the parameterized Cypher query without executing it.",
)
async def preview_query_plan(
    request: RegulatoryQueryRequest,
    service: RegulatoryQueryService = Depends(get_query_service),
) -> Dict[str, Any]:
    """Preview query plan and generated Cypher."""
    try:
        return await service.plan_query(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query planning failed: {str(e)}",
        )


@router.get(
    "/archetypes",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="List supported regulatory query archetypes and sample queries",
)
async def list_query_archetypes() -> List[Dict[str, Any]]:
    """List archetypes and preset demo questions for the dashboard."""
    return [
        {
            "type": QueryType.CURRENT.value,
            "title": "Current Active Regulation",
            "description": "Returns verified active rules, prohibitions, and requirements as of today.",
            "sample_queries": [
                "What are the current client margin reporting requirements for Trading Members?",
                "Is a Trading Member permitted to carry on business other than securities?",
                "What are the active compliance duties of Depository Participants?",
            ],
        },
        {
            "type": QueryType.HISTORICAL.value,
            "title": "Historical Point-in-Time",
            "description": "Returns the regulatory state that was legally effective at a specific date.",
            "sample_queries": [
                "What were the margin reporting rules on 2022-01-07?",
                "What was the valid regulation for Trading Members as of 2022-06-01?",
            ],
        },
        {
            "type": QueryType.EVOLUTION.value,
            "title": "Temporal Rule Evolution",
            "description": "Traverses modification, supersession, and clarification chains across multiple circulars.",
            "sample_queries": [
                "How has Rule 8(1)(f) on Trading Member business evolved over time?",
                "Show the evolution of margin reporting requirements between 2022 and 2026.",
            ],
        },
        {
            "type": QueryType.AMENDMENT.value,
            "title": "Circular Amendment Delta",
            "description": "Inspects specific deltas, modified clauses, and deleted points introduced by an amending circular.",
            "sample_queries": [
                "What changes were introduced by Circular 2026/amendment?",
                "What points were deleted or modified in Circular 50957?",
            ],
        },
        {
            "type": QueryType.MASTER_TO_LATEST.value,
            "title": "Master to Latest Lineage",
            "description": "Traces full lineage from origin/master circular through all subsequent amendment chains.",
            "sample_queries": [
                "Trace Master Circular 2022 to the latest active state.",
                "Show the full amendment lineage of Circular 50957.",
            ],
        },
    ]


@router.get(
    "/diagnostics",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Neo4j graph construction diagnostics and counts",
    description="Returns live counts for Documents, Chunks, Facts, Active Facts, Entities, and Evolution relationships.",
)
async def get_graph_diagnostics() -> Dict[str, Any]:
    """Get live graph diagnostics and count statistics."""
    repo = GraphRepository(neo4j_client)
    return repo.get_graph_diagnostics()


