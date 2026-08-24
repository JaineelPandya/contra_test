"""Dead-letter and retryable job models for failed LLM requests."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Type of LLM task that failed."""
    FACT_EXTRACTION = "FACT_EXTRACTION"
    RELATIONSHIP_CLASSIFICATION = "RELATIONSHIP_CLASSIFICATION"
    METADATA_EXTRACTION = "METADATA_EXTRACTION"


class JobStatus(str, Enum):
    """Lifecycle status of a dead-letter job."""
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    REPROCESSED_SUCCESS = "REPROCESSED_SUCCESS"
    PERMANENTLY_FAILED = "PERMANENTLY_FAILED"


class DeadLetterJob(BaseModel):
    """Failed LLM job preserved with complete payload for background reprocessing."""
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    job_type: JobType
    status: JobStatus = JobStatus.FAILED_RETRYABLE
    attempt_count: int = 1
    error_code: Optional[int] = None
    error_message: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
