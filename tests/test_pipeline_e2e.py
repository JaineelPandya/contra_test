"""End-to-end pipeline verification test simulating the 2022 and 2026 circular workflow."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from app.llm.mock_provider import MockProvider
from app.models.change import ChangeEvent, ChangeType
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus, ProcessingStage
from app.models.entity import Entity
from app.models.fact import Fact, FactStatus
from app.models.hierarchy import HierarchyNode
from app.models.reference import DocumentReference
from app.services.pipeline import DocumentPipeline


class InMemoryRepository:
    """In-memory mock repository simulating Neo4j storage for end-to-end testing."""

    def __init__(self) -> None:
        self.documents: Dict[str, Document] = {}
        self.hierarchy: Dict[str, List[HierarchyNode]] = {}
        self.chunks: Dict[str, List[Chunk]] = {}
        self.facts: Dict[str, List[Fact]] = {}
        self.entities: Dict[str, Entity] = {}
        self.references: List[DocumentReference] = []
        self.changes: List[ChangeEvent] = []

    def save_document(self, doc: Document) -> None:
        self.documents[doc.document_id] = doc

    def update_document_status(
        self,
        document_id: str,
        status: Optional[ProcessingStatus] = None,
        stage: Optional[ProcessingStage] = None,
        pages_processed: Optional[int] = None,
        total_pages: Optional[int] = None,
        sections_created: Optional[int] = None,
        chunks_created: Optional[int] = None,
        facts_extracted: Optional[int] = None,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if document_id in self.documents:
            doc = self.documents[document_id]
            if status:
                doc.status = status
            if stage:
                doc.current_stage = stage
            if pages_processed is not None:
                doc.pages_processed = pages_processed
            if total_pages is not None:
                doc.total_pages = total_pages
            if sections_created is not None:
                doc.sections_created = sections_created
            if chunks_created is not None:
                doc.chunks_created = chunks_created
            if facts_extracted is not None:
                doc.facts_extracted = facts_extracted
            if errors is not None:
                doc.errors = errors
            if metadata:
                for k, v in metadata.items():
                    if hasattr(doc, k) and v is not None:
                        setattr(doc, k, v)

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id)

    def list_documents(self) -> List[Document]:
        return list(self.documents.values())

    def find_document_by_reference(
        self,
        reference_number: Optional[str] = None,
        circular_number: Optional[str] = None,
        date: Optional[str] = None,
        exclude_document_id: Optional[str] = None,
    ) -> Optional[Document]:
        for doc in self.documents.values():
            if exclude_document_id and doc.document_id == exclude_document_id:
                continue
            if reference_number and doc.reference_number and reference_number.lower() in doc.reference_number.lower():
                return doc
            if circular_number and doc.circular_number == circular_number:
                return doc
        return None

    def save_hierarchy_nodes(self, document_id: str, nodes: List[HierarchyNode]) -> None:
        self.hierarchy[document_id] = nodes

    def get_document_hierarchy(self, document_id: str) -> List[Dict[str, Any]]:
        nodes = self.hierarchy.get(document_id, [])
        return [n.model_dump() for n in nodes]

    def save_chunks(self, document_id: str, chunks: List[Chunk]) -> None:
        self.chunks[document_id] = chunks

    def save_entities(self, entities: List[Entity]) -> None:
        for e in entities:
            self.entities[e.normalized_name] = e

    def save_facts(self, document_id: str, facts: List[Fact]) -> None:
        self.facts[document_id] = facts

    def get_document_facts(self, document_id: str) -> List[Dict[str, Any]]:
        facts = self.facts.get(document_id, [])
        return [f.model_dump() for f in facts]

    def save_reference(self, ref: DocumentReference) -> None:
        self.references.append(ref)

    def save_change_event(self, event: ChangeEvent) -> None:
        self.changes.append(event)

    def update_fact_temporal_status(self, fact_id: str, status: FactStatus, valid_to: Optional[str] = None) -> None:
        for doc_facts in self.facts.values():
            for f in doc_facts:
                if f.fact_id == fact_id:
                    f.status = status
                    if valid_to:
                        f.valid_to = valid_to

    def find_target_facts_by_point_or_clause(self, target_document_id: str, point_or_clause: str) -> List[Dict[str, Any]]:
        matched = []
        doc_facts = self.facts.get(target_document_id, [])
        for f in doc_facts:
            if point_or_clause.lower() in f.source_text.lower() or point_or_clause.lower() in f.subject.lower():
                matched.append({
                    "fact_id": f.fact_id,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "status": f.status,
                    "clause_title": point_or_clause,
                })
        return matched

    def get_document_changes(self, document_id: str) -> List[Dict[str, Any]]:
        return [c.model_dump() for c in self.changes if c.source_document_id == document_id or c.target_document_id == document_id]

    def update_facts_status_by_clause(
        self,
        document_id: str,
        clause_identifier: str,
        new_status: FactStatus,
        valid_to: Optional[str] = None,
        change_event_id: Optional[str] = None,
    ) -> int:
        count = 0
        doc_facts = self.facts.get(document_id, [])
        for f in doc_facts:
            if clause_identifier.lower() in f.source_text.lower() or clause_identifier.lower() in f.object.lower():
                f.status = new_status
                if valid_to:
                    f.valid_to = valid_to
                count += 1
        return count

    def create_fact_relationship(self, source_fact_id: str, target_fact_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> None:
        pass

    def save_dead_letter_job(self, job_dict: Dict[str, Any]) -> None:
        pass


@pytest.mark.asyncio
async def test_end_to_end_two_circular_workflow(sample_pdf_dir: Path, mock_llm_provider: MockProvider) -> None:
    """Test the complete workflow:
    1. Upload and process 2022 Circular (12 illustrative activities)
    2. Upload and process 2026 Amending Circular (modifies Points 1 & 2, deletes Points 3-12)
    3. Assert cross-document reference, change events, and temporal state evolution.
    """
    repo = InMemoryRepository()
    pipeline = DocumentPipeline(repository=repo, llm_provider=mock_llm_provider)

    # 1. PROCESS 2022 CIRCULAR
    doc1_id = "doc-2022-uuid"
    doc1_path = sample_pdf_dir / "circular_2022_50957.pdf"
    repo.save_document(
        Document(
            document_id=doc1_id,
            filename="circular_2022_50957.pdf",
            file_path=str(doc1_path),
            status=ProcessingStatus.PROCESSING,
        )
    )
    await pipeline.process_document(doc1_id, doc1_path)

    doc1 = repo.get_document(doc1_id)
    assert doc1.status == ProcessingStatus.COMPLETED
    assert doc1.reference_number == "NSE/COMP/50957"
    assert doc1.document_date == "2022-01-07"
    assert len(repo.facts[doc1_id]) >= 12

    # 2. PROCESS 2026 AMENDING CIRCULAR
    doc2_id = "doc-2026-uuid"
    doc2_path = sample_pdf_dir / "circular_2026_amendment.pdf"
    repo.save_document(
        Document(
            document_id=doc2_id,
            filename="circular_2026_amendment.pdf",
            file_path=str(doc2_path),
            status=ProcessingStatus.PROCESSING,
        )
    )
    await pipeline.process_document(doc2_id, doc2_path)

    doc2 = repo.get_document(doc2_id)
    assert doc2.status == ProcessingStatus.COMPLETED
    assert doc2.reference_number == "NSE/COMP/62114"
    assert doc2.document_date == "2026-06-23"

    # 3. VERIFY CROSS-DOCUMENT REFERENCE
    doc2_refs = [r for r in repo.references if r.source_document_id == doc2_id]
    assert len(doc2_refs) >= 1
    ref = doc2_refs[0]
    assert ref.source_document_id == doc2_id
    assert ref.target_document_id == doc1_id

    # 4. VERIFY CHANGE EVENTS & TEMPORAL EVOLUTION
    changes = repo.get_document_changes(doc2_id)
    assert len(changes) >= 12  # 2 modifications + 10 deletions

    # Check deletions for points 3 to 12
    deleted_points = [c["target_point_or_clause"] for c in changes if c["change_type"] == "DELETED"]
    for p_num in range(3, 13):
        assert f"Point {p_num}" in deleted_points

    # Check modifications for points 1 & 2
    modified_points = [c["target_point_or_clause"] for c in changes if c["change_type"] == "MODIFIED"]
    assert "Point 1" in modified_points
    assert "Point 2" in modified_points

    # Check that deleted facts in Doc 1 have updated status to DELETED and valid_to = 2026-06-23
    doc1_facts = repo.facts[doc1_id]
    for fact in doc1_facts:
        if any(f"Point {p}" in fact.source_text for p in range(3, 13)):
            assert fact.status == FactStatus.DELETED
            assert fact.valid_to == "2026-06-23"
