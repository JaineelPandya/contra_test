"""Hierarchical chunk models."""
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Hierarchical chunk retaining path provenance and sequence pointers."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    heading_path: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    page_start: int = 1
    page_end: int = 1
    order: int = 1
    token_count: int = 0
