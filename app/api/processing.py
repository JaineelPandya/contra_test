"""Processing status API router."""
from fastapi import APIRouter, HTTPException, status
from app.graph.neo4j_client import neo4j_client
from app.graph.repository import GraphRepository
from app.models.document import DocumentStatusResponse

router = APIRouter(prefix="/api/documents", tags=["Processing Status"])


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_processing_status(document_id: str) -> DocumentStatusResponse:
    """Get the current processing status and stage metrics for a document."""
    repo = GraphRepository(neo4j_client)
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    return DocumentStatusResponse(
        document_id=doc.document_id,
        status=doc.status,
        current_stage=doc.current_stage,
        pages_processed=doc.pages_processed,
        total_pages=doc.total_pages,
        sections_created=doc.sections_created,
        chunks_created=doc.chunks_created,
        facts_extracted=doc.facts_extracted,
        errors=doc.errors,
    )
