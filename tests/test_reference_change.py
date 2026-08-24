"""Unit tests for reference detection, normalization, and change classification."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from app.llm.mock_provider import MockProvider
from app.models.change import ChangeType
from app.models.document import Document
from app.models.fact import Fact, FactStatus
from app.models.reference import DocumentReference
from app.services.change_detector import ChangeDetector
from app.services.chunker import HierarchicalChunker
from app.services.hierarchy_builder import HierarchyBuilder
from app.services.pdf_parser import PDFParser
from app.services.reference_detector import ReferenceDetector, normalize_reference_number


def test_normalize_reference_number() -> None:
    assert normalize_reference_number("NSE/COMP/50957") == "NSE/COMP/50957"
    assert normalize_reference_number("NSE / COMP / 50957") == "NSE/COMP/50957"
    assert normalize_reference_number("nse/comp/50957") == "NSE/COMP/50957"
    assert normalize_reference_number("NSE/COMP50957") == "NSE/COMP/50957"
    assert normalize_reference_number("NSE-COMP-50957") == "NSE/COMP/50957"
    assert normalize_reference_number("COMP50957") == "COMP50957"
    assert normalize_reference_number(None) is None


@pytest.mark.asyncio
async def test_reference_detector_2026_to_2022(sample_pdf_dir: Path, mock_llm_provider: MockProvider) -> None:
    pdf_path = sample_pdf_dir / "circular_2026_amendment.pdf"
    pages, _ = PDFParser.parse_pdf(pdf_path)

    builder = HierarchyBuilder()
    nodes = builder.build_hierarchy("doc-2026", pages)
    chunker = HierarchicalChunker()
    chunks = chunker.create_chunks("doc-2026", nodes)

    mock_repo = MagicMock()
    mock_repo.find_document_by_reference.return_value = Document(
        document_id="doc-2022-uuid",
        filename="circular_2022_50957.pdf",
        file_path="/tmp/2022.pdf",
        reference_number="NSE/COMP/50957",
        circular_number="02/2022",
        document_date="2022-01-07",
    )

    detector = ReferenceDetector(mock_repo, mock_llm_provider)
    refs = await detector.detect_references("doc-2026", chunks)

    assert len(refs) >= 1
    matched = [r for r in refs if r.target_reference_number == "NSE/COMP/50957" or r.target_document_id == "doc-2022-uuid"]
    assert len(matched) >= 1
    assert matched[0].target_document_id == "doc-2022-uuid"
    assert mock_repo.save_reference.called


@pytest.mark.asyncio
async def test_reference_detector_insp74836_text(mock_llm_provider: MockProvider) -> None:
    insp_text = """
    NATIONAL STOCK EXCHANGE OF INDIA LIMITED
    Circular Ref. No: 32/2026
    Download Ref No: NSE/INSP/74836
    Date: June 23, 2026

    To All Members,
    Sub: Clarification to Rule 8(1)(f) and 8(3)(f) of Securities Contracts (Regulation) Rules, 1957

    In view of the above amendment, Point No. 1 and 2 of illustrative list of activities issued vide Exchange
    Circular Ref No. NSE/COMP/50957 dated January 07, 2022 which are construed as non-compliance to Rule
    8(1)(f) and 8(3)(f) of SCRR, if undertaken by a member, are modified as below: -

    1. Issuing Corporate Guarantees towards credit facilities availed by any entity.
    2. Deposit pledged with the bank for overdraft facilities.

    Further, Point No. 3 to 12 of illustrative list of activities as prescribed in Exchange Circular Ref No.
    NSE/COMP/50957 dated January 07, 2022, stand deleted.
    """
    chunk = MagicMock()
    chunk.content = insp_text

    mock_repo = MagicMock()
    mock_repo.find_document_by_reference.return_value = Document(
        document_id="comp-50957-uuid",
        filename="COMP50957.pdf",
        file_path="/uploads/COMP50957.pdf",
        reference_number="NSE/COMP/50957",
        circular_number="02/2022",
        document_date="2022-01-07",
    )

    detector = ReferenceDetector(mock_repo, mock_llm_provider)
    refs = await detector.detect_references("insp-74836-uuid", [chunk])

    assert len(refs) >= 1
    matched = [r for r in refs if r.target_reference_number == "NSE/COMP/50957"]
    assert len(matched) >= 1
    assert matched[0].target_document_id == "comp-50957-uuid"


@pytest.mark.asyncio
async def test_change_detector_insp74836_changes(mock_llm_provider: MockProvider) -> None:
    insp_text = """
    In view of the above amendment, Point No. 1 and 2 of illustrative list of activities issued vide Exchange
    Circular Ref No. NSE/COMP/50957 dated January 07, 2022 which are construed as non-compliance to Rule
    8(1)(f) and 8(3)(f) of SCRR, if undertaken by a member, are modified as below: -

    1. Issuing Corporate Guarantees towards credit facilities availed by any entity.
    2. Deposit pledged with the bank for overdraft facilities.

    Further, Point No. 3 to 12 of illustrative list of activities as prescribed in Exchange Circular Ref No.
    NSE/COMP/50957 dated January 07, 2022, stand deleted.
    """
    mock_repo = MagicMock()
    detector = ChangeDetector(mock_repo, mock_llm_provider)

    changes = detector._detect_deterministic_changes(
        source_doc_id="insp-74836",
        target_doc_id="comp-50957",
        text=insp_text,
        effective_date="2026-06-23",
    )

    mod_changes = [c for c in changes if c.change_type == ChangeType.MODIFIED]
    assert len(mod_changes) >= 2
    assert any(c.target_point_or_clause == "Point 1" for c in mod_changes)
    assert any(c.target_point_or_clause == "Point 2" for c in mod_changes)

    del_changes = [c for c in changes if c.change_type == ChangeType.DELETED]
    assert len(del_changes) == 10
    for p_num in range(3, 13):
        assert any(c.target_point_or_clause == f"Point {p_num}" for c in del_changes)


@pytest.mark.asyncio
async def test_change_detector_range_deletion(sample_pdf_dir: Path, mock_llm_provider: MockProvider) -> None:
    pdf_path = sample_pdf_dir / "circular_2026_amendment.pdf"
    pages, _ = PDFParser.parse_pdf(pdf_path)

    builder = HierarchyBuilder()
    nodes = builder.build_hierarchy("doc-2026", pages)
    chunker = HierarchicalChunker()
    chunks = chunker.create_chunks("doc-2026", nodes)

    mock_repo = MagicMock()
    detector = ChangeDetector(mock_repo, mock_llm_provider)

    changes = detector._detect_deterministic_changes(
        source_doc_id="doc-2026",
        target_doc_id="doc-2022",
        text=PDFParser.get_full_text(pages),
        effective_date="2026-06-23",
    )

    mod_changes = [c for c in changes if c.change_type == ChangeType.MODIFIED]
    assert len(mod_changes) >= 2
    assert any(c.target_point_or_clause == "Point 1" for c in mod_changes)
    assert any(c.target_point_or_clause == "Point 2" for c in mod_changes)

    del_changes = [c for c in changes if c.change_type == ChangeType.DELETED]
    assert len(del_changes) == 10
    for p_num in range(3, 13):
        assert any(c.target_point_or_clause == f"Point {p_num}" for c in del_changes)
