"""Atomic fact and temporal status models."""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class FactPredicate(str, Enum):
    """Canonical standardized predicate categories for regulatory facts and evolution events.

    Regulatory & Obligation Predicates:
      - REQUIRES: Mandatory obligation or compliance action
      - PROHIBITS: Explicit restriction or barred activity
      - PERMITS: Authorization, exemption entitlement, or allowed discretion
      - DEFINES: Term definition, threshold criteria, or scope meaning
      - APPLIES_TO: Scope of applicability to classes or instruments
      - SUBJECT_TO: Conditional dependency on another rule/clause
      - EXCEPTION_TO: Carve-out, exemption, or non-applicability provision
      - VIOLATES: Action constituting non-compliance
      - RELATED_TO: Topical association between provisions
      - REFERENCES: Non-amendment citation or reference to another document/clause

    Evolution & Change Predicates:
      - AMENDS: Substantive modification of an existing provision
      - MODIFIES: Direct alteration or adjustment to a rule/threshold
      - DELETES: Deletion or omission of a clause/rule
      - REPLACES: Complete substitution of an existing provision with new text
      - SUPERSEDES: Full replacement or displacement of an earlier circular/rule
      - CLARIFIES: Explanatory clarification or FAQ interpretation
      - EXTENDS: Extension of deadline, timeline, or scope
      - REPEALS: Formal revocation or termination of a regulatory instrument
    """
    # Regulatory & Obligation Predicates
    REQUIRES = "REQUIRES"
    PROHIBITS = "PROHIBITS"
    PERMITS = "PERMITS"
    DEFINES = "DEFINES"
    APPLIES_TO = "APPLIES_TO"
    SUBJECT_TO = "SUBJECT_TO"
    EXCEPTION_TO = "EXCEPTION_TO"
    VIOLATES = "VIOLATES"
    RELATED_TO = "RELATED_TO"
    REFERENCES = "REFERENCES"

    # Granular Regulatory & Procedural Predicates
    CREDITED_TO = "CREDITED_TO"
    PAYMENT_DEADLINE = "PAYMENT_DEADLINE"
    SUBMITS_TO = "SUBMITS_TO"
    REPORTS_TO = "REPORTS_TO"
    MAINTAINS = "MAINTAINS"
    IMPLEMENTS = "IMPLEMENTS"

    # Evolution & Change Predicates
    AMENDS = "AMENDS"
    MODIFIES = "MODIFIES"
    DELETES = "DELETES"
    REPLACES = "REPLACES"
    SUPERSEDES = "SUPERSEDES"
    CLARIFIES = "CLARIFIES"
    EXTENDS = "EXTENDS"
    REPEALS = "REPEALS"


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace, line breaks, and lowercase for grounding validation."""
    if not text:
        return ""
    return " ".join(text.lower().split())


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
    page_number: Optional[int] = Field(None, description="Source page number in the original document")

    subject: str = Field(..., description="Entity or agent subject of the fact")
    predicate: FactPredicate = Field(..., description="Action/relationship category")
    object: str = Field(..., min_length=1, description="Target action, requirement, or entity (MUST NOT BE EMPTY)")

    rule_reference: Optional[str] = Field(None, description="Anchor rule e.g. 'Rule 8(1)(f)', 'Rule 8(3)(f)'")
    clause_reference: Optional[str] = Field(None, description="Anchor clause or point e.g. 'Point 1', 'Clause (a)'")

    conditions: List[str] = Field(default_factory=list, description="Conditions required for this fact")
    exceptions: List[str] = Field(default_factory=list, description="Exceptions where fact does not apply")
    threshold: Optional[str] = Field(None, description="Quantified threshold limit e.g. '10 Lakhs in a calendar quarter'")
    time: Optional[str] = Field(None, description="Time limit or deadline e.g. '15 working days', 'immediately'")
    source_circular: Optional[str] = Field(None, description="Citing source circular reference")

    regulatory_status: str = Field("ACTIVE", description="Regulatory requirement state: PROHIBITED, PERMITTED, MANDATORY, etc.")
    effective_date: Optional[str] = Field(None, description="Effective date if mentioned in text")
    confidence: float = Field(1.0, description="Extraction confidence score (0.0 - 1.0)")
    source_text: str = Field(..., description="Exact supporting verbatim sentence from text")

    # Provenance & Method Tracking
    extraction_method: str = Field("groq", description="'groq', 'gemini', 'openai', or 'rule_based'")
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
