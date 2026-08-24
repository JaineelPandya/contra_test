"""Atomic fact and temporal status models."""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class FactPredicate(str, Enum):
    """Standardized predicate categories for regulatory facts."""
    REQUIRES = "REQUIRES"
    PROHIBITS = "PROHIBITS"
    PERMITS = "PERMITS"
    DEFINES = "DEFINES"
    APPLIES_TO = "APPLIES_TO"
    AMENDS = "AMENDS"
    MODIFIES = "MODIFIES"
    DELETES = "DELETES"
    SUPERSEDES = "SUPERSEDES"
    CLARIFIES = "CLARIFIES"
    REFERENCES = "REFERENCES"
    VIOLATES = "VIOLATES"
    SUBJECT_TO = "SUBJECT_TO"
    EXCEPTION_TO = "EXCEPTION_TO"


class FactStatus(str, Enum):
    """Temporal validity status of a fact."""
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"
    MODIFIED = "MODIFIED"
    SUPERSEDED = "SUPERSEDED"
    CLARIFIED = "CLARIFIED"
    INACTIVE = "INACTIVE"


class Fact(BaseModel):
    """Structured atomic fact extracted from text."""
    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_id: str
    hierarchy_node_id: Optional[str] = None

    subject: str = Field(..., description="Entity or agent subject of the fact")
    predicate: FactPredicate = Field(..., description="Action/relationship category")
    object: str = Field(..., description="Target action, requirement, or entity")

    conditions: List[str] = Field(default_factory=list, description="Conditions required for this fact")
    exceptions: List[str] = Field(default_factory=list, description="Exceptions where fact does not apply")

    regulatory_status: str = Field("ACTIVE", description="Regulatory requirement state: PROHIBITED, PERMITTED, MANDATORY, etc.")
    effective_date: Optional[str] = Field(None, description="Effective date if mentioned in text")
    confidence: float = Field(1.0, description="Extraction confidence score (0.0 - 1.0)")
    source_text: str = Field(..., description="Exact supporting verbatim sentence from text")

    # Provenance & Method Tracking
    extraction_method: str = Field("gemini", description="'gemini', 'openai', 'ollama', or 'rule_based'")
    model: Optional[str] = Field(None, description="LLM model name if extracted via model")
    fallback_reason: Optional[str] = Field(None, description="Reason if rule-based fallback was used")

    # Temporal Tracking Fields
    status: FactStatus = Field(FactStatus.ACTIVE, description="Current lifecycle state of the fact")
    valid_from: Optional[str] = Field(None, description="Start date of validity (e.g. document date)")
    valid_to: Optional[str] = Field(None, description="End date of validity if superseded or deleted")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FactExtractionResult(BaseModel):
    """Container for facts extracted by LLM from a chunk."""
    facts: List[Fact] = Field(default_factory=list)
