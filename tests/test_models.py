"""Unit tests for Pydantic models and schemas."""
import pytest
from app.models.document import Document, DocumentMetadata, ProcessingStatus, ProcessingStage
from app.models.hierarchy import HierarchyNode, HierarchyNodeType
from app.models.chunk import Chunk
from app.models.fact import Fact, FactPredicate, FactStatus
from app.models.entity import Entity, EntityType
from app.models.reference import DocumentReference
from app.models.change import ChangeEvent, ChangeType


def test_document_model_defaults() -> None:
    doc = Document(filename="test.pdf", file_path="/tmp/test.pdf")
    assert doc.document_id is not None
    assert doc.status == ProcessingStatus.UPLOADED
    assert doc.current_stage == ProcessingStage.UPLOADED
    assert doc.errors == []


def test_hierarchy_tree_nesting() -> None:
    parent = HierarchyNode(
        document_id="doc-123",
        node_type=HierarchyNodeType.SECTION,
        title="Annexure A",
        content="Annexure A text",
        level=1,
        order=1,
    )
    child = HierarchyNode(
        document_id="doc-123",
        node_type=HierarchyNodeType.POINT,
        title="Point 1",
        content="Point 1 content",
        level=2,
        order=2,
        parent_id=parent.id,
    )
    parent.children.append(child)
    assert len(parent.children) == 1
    assert parent.children[0].parent_id == parent.id


def test_fact_model_predicates_and_temporal() -> None:
    fact = Fact(
        document_id="doc-123",
        chunk_id="chunk-456",
        subject="Trading Member",
        predicate=FactPredicate.PROHIBITS,
        object="Financing securities transactions",
        source_text="Financing securities transactions is prohibited.",
        status=FactStatus.ACTIVE,
        valid_from="2022-01-07",
    )
    assert fact.predicate == FactPredicate.PROHIBITS
    assert fact.status == FactStatus.ACTIVE
    assert fact.valid_from == "2022-01-07"
    assert fact.valid_to is None

    # Serialization test
    dump = fact.model_dump()
    assert dump["predicate"] == "PROHIBITS"
    assert dump["status"] == "ACTIVE"


def test_change_event_model() -> None:
    event = ChangeEvent(
        source_document_id="doc-2026",
        target_document_id="doc-2022",
        change_type=ChangeType.DELETED,
        target_point_or_clause="Point 3",
        description="Point 3 stands deleted",
        effective_date="2026-06-23",
    )
    assert event.change_type == ChangeType.DELETED
    assert event.target_point_or_clause == "Point 3"
