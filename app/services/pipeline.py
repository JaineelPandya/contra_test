"""Asynchronous processing pipeline orchestrating all extraction stages."""
from pathlib import Path
from typing import List, Optional
from app.core.logging import logger
from app.graph.repository import GraphRepository
from app.llm.base import BaseLLMProvider
from app.models.document import ProcessingStatus, ProcessingStage
from app.services.pdf_parser import PDFParser, PDFPage
from app.services.metadata_extractor import MetadataExtractor
from app.services.hierarchy_builder import HierarchyBuilder
from app.services.chunker import HierarchicalChunker
from app.services.fact_extractor import FactExtractor
from app.services.entity_resolver import EntityResolver
from app.services.reference_detector import ReferenceDetector
from app.services.change_detector import ChangeDetector


class DocumentPipeline:
    """Orchestrates end-to-end processing of a regulatory PDF into the Neo4j knowledge graph."""

    def __init__(self, repository: GraphRepository, llm_provider: BaseLLMProvider) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.pdf_parser = PDFParser()
        self.metadata_extractor = MetadataExtractor(llm_provider)
        self.hierarchy_builder = HierarchyBuilder()
        self.chunker = HierarchicalChunker()
        self.fact_extractor = FactExtractor(llm_provider)
        self.entity_resolver = EntityResolver(llm_provider)
        self.reference_detector = ReferenceDetector(repository, llm_provider)
        self.change_detector = ChangeDetector(repository, llm_provider)

    async def process_document(self, document_id: str, file_path: Path) -> None:
        """Run the end-to-end extraction pipeline asynchronously."""
        logger.info("Starting processing pipeline for document_id: %s (file: %s)", document_id, file_path)
        errors: List[str] = []

        try:
            # 1. TEXT EXTRACTION
            self.repository.update_document_status(
                document_id,
                status=ProcessingStatus.PROCESSING,
                stage=ProcessingStage.TEXT_EXTRACTION,
            )
            pages, total_pages = self.pdf_parser.parse_pdf(file_path)
            self.repository.update_document_status(
                document_id,
                pages_processed=total_pages,
                total_pages=total_pages,
            )

            # 2. METADATA EXTRACTION
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.METADATA_EXTRACTION,
            )
            metadata = await self.metadata_extractor.extract_metadata(pages, total_pages)
            self.repository.update_document_status(
                document_id,
                metadata={
                    "source": metadata.source,
                    "department": metadata.department,
                    "reference_number": metadata.reference_number,
                    "circular_number": metadata.circular_number,
                    "document_date": metadata.document_date,
                    "title": metadata.title,
                },
            )

            # 3. HIERARCHY BUILDING
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.HIERARCHY_BUILDING,
            )
            hierarchy_nodes = self.hierarchy_builder.build_hierarchy(document_id, pages)
            self.repository.update_document_status(
                document_id,
                sections_created=len(hierarchy_nodes),
            )

            # 4. HIERARCHICAL CHUNKING
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.CHUNKING,
            )
            chunks = self.chunker.create_chunks(document_id, hierarchy_nodes)
            self.repository.update_document_status(
                document_id,
                chunks_created=len(chunks),
            )

            # Dead-letter callback for recording failed LLM requests
            def record_dead_letter(job_type: str, status_code: int, error_msg: str, payload: dict) -> None:
                from app.models.dead_letter import DeadLetterJob, JobType, JobStatus
                job = DeadLetterJob(
                    document_id=document_id,
                    job_type=JobType(job_type),
                    status=JobStatus.FAILED_RETRYABLE,
                    error_code=status_code,
                    error_message=error_msg,
                    payload=payload,
                )
                self.repository.save_dead_letter_job(job.model_dump())
                logger.warning("Recorded dead-letter job for document %s (type=%s, code=%d)", document_id, job_type, status_code)

            # 5. FACT EXTRACTION (Batched)
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.FACT_EXTRACTION,
            )
            facts = await self.fact_extractor.extract_all_facts(
                document_id=document_id,
                chunks=chunks,
                metadata=metadata,
                on_dead_letter=lambda code, err, payload: record_dead_letter("FACT_EXTRACTION", code, err, payload),
            )
            self.repository.update_document_status(
                document_id,
                facts_extracted=len(facts),
            )

            # 6. DETERMINISTIC ENTITY RESOLUTION (Zero LLM calls)
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.ENTITY_RESOLUTION,
            )
            entities = self.entity_resolver.resolve_from_facts(facts=facts, chunks=chunks)

            # 7. NEO4J CORE STORAGE PERSISTENCE (Save hierarchy, chunks, entities, facts before relationship linking)
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.NEO4J_STORAGE,
            )
            self.repository.save_hierarchy_nodes(document_id, hierarchy_nodes)
            self.repository.save_chunks(document_id, chunks)
            self.repository.save_entities(entities)
            self.repository.save_facts(document_id, facts)

            # 8. REFERENCE DETECTION
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.REFERENCE_DETECTION,
            )
            references = await self.reference_detector.detect_references(document_id, chunks)
            for ref in references:
                self.repository.save_reference(ref)

            # 9. CHANGE DETECTION & TEMPORAL EVOLUTION (Candidate Generation + Batched Semantic Verification)
            self.repository.update_document_status(
                document_id,
                stage=ProcessingStage.CHANGE_DETECTION,
            )
            change_events = await self.change_detector.detect_and_apply_changes(
                source_document_id=document_id,
                references=references,
                chunks=chunks,
                new_facts=facts,
                metadata=metadata,
                on_dead_letter=lambda code, err, payload: record_dead_letter("RELATIONSHIP_CLASSIFICATION", code, err, payload),
            )

            # 10. MARK COMPLETED
            self.repository.update_document_status(
                document_id,
                status=ProcessingStatus.COMPLETED,
                stage=ProcessingStage.COMPLETED,
            )
            logger.info("Successfully completed processing pipeline for document_id: %s", document_id)


        except Exception as e:
            error_msg = f"Pipeline execution failed for document {document_id}: {str(e)}"
            logger.exception(error_msg)
            errors.append(error_msg)
            self.repository.update_document_status(
                document_id,
                status=ProcessingStatus.FAILED,
                stage=ProcessingStage.FAILED,
                errors=errors,
            )
