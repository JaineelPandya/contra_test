"""Change detection event models."""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Classification of regulatory change actions."""
    CREATED = "CREATED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    SUPERSEDED = "SUPERSEDED"
    CLARIFIED = "CLARIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNCHANGED = "UNCHANGED"


class ChangeEvent(BaseModel):
    """Represents a discrete change event between regulatory artifacts."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_document_id: str
    target_document_id: Optional[str] = None
    source_chunk_id: Optional[str] = None
    target_clause_id: Optional[str] = None
    source_fact_id: Optional[str] = None
    target_fact_id: Optional[str] = None

    change_type: ChangeType
    target_point_or_clause: Optional[str] = None
    description: str
    effective_date: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
