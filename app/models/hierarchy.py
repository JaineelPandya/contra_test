"""Hierarchy models for structured document parsing."""
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class HierarchyNodeType(str, Enum):
    """Types of nodes within a document hierarchy."""
    DOCUMENT = "DOCUMENT"
    SECTION = "SECTION"
    SUBSECTION = "SUBSECTION"
    CLAUSE = "CLAUSE"
    LIST_ITEM = "LIST_ITEM"
    POINT = "POINT"
    ANNEXURE = "ANNEXURE"
    FAQ = "FAQ"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"


class HierarchyNode(BaseModel):
    """Node in the hierarchical tree representing sections, clauses, points, etc."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    node_type: HierarchyNodeType = HierarchyNodeType.SECTION
    title: str
    content: str
    level: int = 1
    order: int = 1
    page_start: int = 1
    page_end: int = 1
    parent_id: Optional[str] = None
    children: List["HierarchyNode"] = Field(default_factory=list)


# Resolve self-referential Pydantic model
HierarchyNode.model_rebuild()
