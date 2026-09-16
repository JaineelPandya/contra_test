"""Data models for Regulatory Query Analysis, Planning, and Graph Response schemas."""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from app.models.fact import FactStatus


class QueryType(str, Enum):
    """Categorized query archetypes for regulatory graph analysis."""
    CURRENT = "CURRENT"                 # Active regulatory requirements today
    HISTORICAL = "HISTORICAL"           # Point-in-time state as of a specific date
    EVOLUTION = "EVOLUTION"             # How a rule/topic has evolved across amendments
    AMENDMENT = "AMENDMENT"             # What was changed by a specific circular/amendment
    MASTER_TO_LATEST = "MASTER_TO_LATEST" # Trace from base/master circular to current active state
    CHANGELOG = "CHANGELOG"             # List ALL changes/amendments/deletions across all circulars


class RegulatoryQueryRequest(BaseModel):
    """Incoming user natural language query payload."""
    query: str = Field(..., description="Natural language question about regulatory rules or changes")
    query_type_override: Optional[QueryType] = Field(None, description="Optional manual override for query archetype")
    as_of_date: Optional[str] = Field(None, description="Optional point-in-time date (YYYY-MM-DD)")
    circular_number: Optional[str] = Field(None, description="Optional specific circular or reference number")
    entity: Optional[str] = Field(None, description="Optional target entity filter (e.g. Trading Member)")
    include_subgraph: bool = Field(True, description="Whether to include graph visualization nodes and edges")


class RegulatoryQueryPlan(BaseModel):
    """Structured plan output by the Query Analyzer and Graph Planner."""
    query_type: QueryType
    raw_query: str
    entities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list, description="Extracted key topics and search terms")
    predicates: List[str] = Field(default_factory=list)
    target_circulars: List[str] = Field(default_factory=list)
    temporal_scope: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field("", description="Explanation of how the query was classified and planned")


class SubgraphNode(BaseModel):
    """Graph node formatted for frontend visualization (Cytoscape / D3)."""
    id: str
    label: str
    type: str  # Document, Fact, Entity, Chunk, ChangeEvent
    status: Optional[str] = None  # ACTIVE, DELETED, MODIFIED, SUPERSEDED
    group: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class SubgraphEdge(BaseModel):
    """Graph relationship formatted for frontend visualization."""
    id: str
    source: str
    target: str
    type: str  # MODIFIES, DELETES, SUPERSEDES, AMENDS, REFERENCES, SUPPORTED_BY, SUBJECT, OBJECT
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphSubgraph(BaseModel):
    """Subgraph payload for dashboard visualization."""
    nodes: List[SubgraphNode] = Field(default_factory=list)
    edges: List[SubgraphEdge] = Field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0


class CurrentRegulationSummary(BaseModel):
    """Summary of the active regulatory state matching user's question."""
    summary: str
    status: FactStatus = FactStatus.ACTIVE
    active_facts_count: int = 0
    effective_date: Optional[str] = None
    primary_circular: Optional[str] = None
    entities_affected: List[str] = Field(default_factory=list)


class RegulatoryChangeRecord(BaseModel):
    """Single chronological regulatory change / amendment event."""
    date: Optional[str] = None
    change_type: str = "MODIFIED"  # MODIFIED, DELETED, SUPERSEDED, CLARIFIED, CREATED
    amending_circular: Optional[str] = None
    target_point_or_clause: Optional[str] = None
    description: str
    source_fact_id: Optional[str] = None
    target_fact_id: Optional[str] = None
    effective_date: Optional[str] = None


class RegulatorySourceCitation(BaseModel):
    """Verbatim source reference supporting the answer."""
    document_id: Optional[str] = None
    circular_number: Optional[str] = None
    reference_number: Optional[str] = None
    document_date: Optional[str] = None
    clause_path: Optional[str] = None
    source_text: str
    status: str = "ACTIVE"
    confidence: float = 1.0


class RegulatoryQueryResponse(BaseModel):
    """Comprehensive 3-part response matching the regulatory dashboard architecture."""
    query: str
    plan: RegulatoryQueryPlan
    cypher_query: str
    cypher_params: Dict[str, Any] = Field(default_factory=dict)
    
    # 1. Current Active Regulation
    current_regulation: Optional[CurrentRegulationSummary] = None
    
    # 2. How it Changed (Chronological Delta)
    change_history: List[RegulatoryChangeRecord] = Field(default_factory=list)
    
    # 3. Comprehensive LLM Synthesized Answer with Provenance
    llm_answer: str
    sources: List[RegulatorySourceCitation] = Field(default_factory=list)
    
    # 4. Interactive Visual Graph Subgraph
    subgraph: Optional[GraphSubgraph] = None
