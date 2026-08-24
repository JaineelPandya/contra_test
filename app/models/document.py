"""Document models and processing stage definitions."""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """High-level processing status of a document."""
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingStage(str, Enum):
    """Granular stages of the processing pipeline."""
    UPLOADED = "UPLOADED"
    METADATA_EXTRACTION = "METADATA_EXTRACTION"
    TEXT_EXTRACTION = "TEXT_EXTRACTION"
    HIERARCHY_BUILDING = "HIERARCHY_BUILDING"
    CHUNKING = "CHUNKING"
    FACT_EXTRACTION = "FACT_EXTRACTION"
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    REFERENCE_DETECTION = "REFERENCE_DETECTION"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    NEO4J_STORAGE = "NEO4J_STORAGE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentMetadata(BaseModel):
    """Extracted metadata for a regulatory document."""
    source: Optional[str] = Field(None, description="Issuing authority, e.g. NSE, SEBI, RBI")
    department: Optional[str] = Field(None, description="Department, e.g. COMPLIANCE, SURVEILLANCE")
    reference_number: Optional[str] = Field(None, description="Official reference number e.g. NSE/COMP/50957")
    circular_number: Optional[str] = Field(None, description="Circular number e.g. 02/2022")
    document_date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Title or subject of the circular")
    total_pages: int = Field(1, description="Total number of pages in the PDF")


class DocumentCreate(BaseModel):
    """Model for creating a new document record."""
    filename: str
    file_path: str


class Document(BaseModel):
    """Complete representation of a document in RegGraph."""
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    file_path: str
    source: Optional[str] = None
    department: Optional[str] = None
    reference_number: Optional[str] = None
    circular_number: Optional[str] = None
    document_date: Optional[str] = None
    title: Optional[str] = None
    total_pages: int = 1

    status: ProcessingStatus = ProcessingStatus.UPLOADED
    current_stage: ProcessingStage = ProcessingStage.UPLOADED

    pages_processed: int = 0
    sections_created: int = 0
    chunks_created: int = 0
    facts_extracted: int = 0

    errors: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentUploadResponse(BaseModel):
    """Response returned immediately upon document upload."""
    document_id: str
    filename: str
    status: ProcessingStatus


class DocumentStatusResponse(BaseModel):
    """Response returned when polling document processing status."""
    document_id: str
    status: ProcessingStatus
    current_stage: ProcessingStage
    pages_processed: int
    total_pages: int
    sections_created: int
    chunks_created: int
    facts_extracted: int
    errors: List[str] = Field(default_factory=list)
