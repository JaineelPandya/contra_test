"""Document management and graph inspection API router."""
import shutil
from typing import Any, Dict, List
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from app.core.config import get_settings
from app.core.logging import logger
from app.graph.neo4j_client import neo4j_client
from app.graph.repository import GraphRepository
from app.llm.factory import get_llm_provider
from app.models.document import Document, DocumentUploadResponse, ProcessingStatus, ProcessingStage
from app.services.pipeline import DocumentPipeline

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a regulatory PDF and initiate asynchronous knowledge graph processing."""
    # Validate PDF content type / extension
    if not (file.filename and file.filename.lower().endswith(".pdf")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    document_id = str(uuid4())
    settings = get_settings()
    upload_dir = settings.upload_path
    saved_filename = f"{document_id}_{file.filename}"
    file_path = upload_dir / saved_filename

    # Save PDF file to persistent disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save uploaded file %s: %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file to disk.",
        )

    # Create initial Document node in Neo4j with status PROCESSING
    repo = GraphRepository(neo4j_client)
    initial_doc = Document(
        document_id=document_id,
        filename=file.filename,
        file_path=str(file_path),
        status=ProcessingStatus.PROCESSING,
        current_stage=ProcessingStage.UPLOADED,
    )
    repo.save_document(initial_doc)

    # Launch asynchronous pipeline in background task
    llm_provider = get_llm_provider()
    pipeline = DocumentPipeline(repo, llm_provider)
    background_tasks.add_task(pipeline.process_document, document_id, file_path)

    logger.info("Enqueued processing task for document %s (%s)", document_id, file.filename)
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=ProcessingStatus.PROCESSING,
    )


@router.get("", response_model=List[Document])
async def list_documents() -> List[Document]:
    """List all stored documents and their metadata."""
    repo = GraphRepository(neo4j_client)
    return repo.list_documents()


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str) -> Document:
    """Retrieve full details and metadata for a specific document."""
    repo = GraphRepository(neo4j_client)
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return doc


@router.get("/{document_id}/hierarchy")
async def get_document_hierarchy(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve the structural hierarchy nodes (Sections, Clauses, FAQs) for a document."""
    repo = GraphRepository(neo4j_client)
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return repo.get_document_hierarchy(document_id)


@router.get("/{document_id}/facts")
async def get_document_facts(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all extracted atomic facts, temporal status, and chunk provenance for a document."""
    repo = GraphRepository(neo4j_client)
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return repo.get_document_facts(document_id)


@router.get("/{document_id}/changes")
async def get_document_changes(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all regulatory change events affecting or initiated by this document."""
    repo = GraphRepository(neo4j_client)
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return repo.get_document_changes(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(document_id: str) -> Dict[str, str]:
    """Delete a document and its entire graph hierarchy, chunks, and facts."""
    repo = GraphRepository(neo4j_client)
    deleted = repo.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return {"message": f"Document '{document_id}' and its related graph nodes deleted successfully."}


@router.get("/{document_id}/failed-jobs", response_model=List[Dict[str, Any]])
async def get_failed_jobs(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all failed retryable LLM jobs recorded in dead-letter store."""
    repo = GraphRepository(neo4j_client)
    return repo.get_dead_letter_jobs(document_id)


@router.post("/{document_id}/reprocess-failed")
async def reprocess_failed_jobs(document_id: str) -> Dict[str, Any]:
    """Reprocess all failed retryable jobs for a document without re-parsing the PDF."""
    repo = GraphRepository(neo4j_client)
    jobs = repo.get_dead_letter_jobs(document_id)
    retryable = [j for j in jobs if j.get("status") == "FAILED_RETRYABLE"]
    if not retryable:
        return {"message": "No failed retryable jobs found for this document.", "reprocessed_count": 0}

    provider = get_llm_provider()
    doc = repo.get_document(document_id)
    reprocessed = 0

    for job in retryable:
        job_id = job["job_id"]
        job_type = job["job_type"]
        payload = job.get("payload", {})

        try:
            if job_type == "FACT_EXTRACTION":
                from app.models.chunk import Chunk
                chunks_raw = payload.get("chunks", [])
                chunks = [
                    Chunk(
                        chunk_id=c["chunk_id"],
                        document_id=document_id,
                        content=c["text"],
                        heading_path=c.get("heading_path", "").split(" > "),
                        token_count=len(c["text"].split()),
                        page_start=1,
                        page_end=1,
                        order=0,
                    )
                    for c in chunks_raw
                ]
                facts = await provider.extract_facts_batch(document_id, chunks, doc.metadata if doc else None)
                if facts:
                    repo.save_facts(document_id, facts)
                    repo.update_dead_letter_job_status(job_id, "REPROCESSED_SUCCESS")
                    reprocessed += 1

            elif job_type == "RELATIONSHIP_CLASSIFICATION":
                pairs = payload.get("candidate_pairs", [])
                results = await provider.classify_relationships_batch(pairs)
                if results:
                    for r in results:
                        rel = r.get("relationship", "NONE")
                        if rel != "NONE" and r.get("source_id") and r.get("target_id"):
                            repo.create_fact_relationship(
                                source_fact_id=r["source_id"],
                                target_fact_id=r["target_id"],
                                relationship_type=rel,
                                properties=r,
                            )
                    repo.update_dead_letter_job_status(job_id, "REPROCESSED_SUCCESS")
                    reprocessed += 1

        except Exception as exc:
            repo.update_dead_letter_job_status(job_id, "FAILED_RETRYABLE")

    return {
        "message": f"Reprocessed {reprocessed}/{len(retryable)} failed jobs.",
        "reprocessed_count": reprocessed,
        "total_jobs": len(retryable),
    }

