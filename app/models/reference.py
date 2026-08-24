"""Reference models for cross-document citations and links."""
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class DocumentReference(BaseModel):
    """Citation or reference from one document to another (or to a specific section/point)."""
    reference_id: str = Field(default_factory=lambda: str(uuid4()))
    source_document_id: str
    target_document_id: Optional[str] = None
    source_clause_id: Optional[str] = None
    target_clause_id: Optional[str] = None

    # Reference metadata extracted from text
    target_reference_number: Optional[str] = Field(None, description="e.g. NSE/COMP/50957")
    target_circular_number: Optional[str] = Field(None, description="e.g. 02/2022")
    target_date: Optional[str] = Field(None, description="e.g. 2022-01-07")
    target_point_or_clause: Optional[str] = Field(None, description="e.g. Point No. 3, Rule 8(1)(f), Clause 20")

    reference_text: str = Field(..., description="Verbatim text containing the citation")
    reference_type: str = Field("REFERENCES", description="REFERENCES, AMENDS, CLARIFIES, SUPERSEDES, etc.")
    confidence: float = Field(1.0, description="Match confidence")
