"""Unit tests for hierarchy builder and hierarchical chunker."""
from pathlib import Path
import pytest
from app.models.hierarchy import HierarchyNodeType
from app.services.chunker import HierarchicalChunker
from app.services.hierarchy_builder import HierarchyBuilder
from app.services.pdf_parser import PDFParser


def test_hierarchy_builder_2022_circular(sample_pdf_dir: Path) -> None:
    pdf_path = sample_pdf_dir / "circular_2022_50957.pdf"
    pages, _ = PDFParser.parse_pdf(pdf_path)

    builder = HierarchyBuilder()
    nodes = builder.build_hierarchy("doc-2022", pages)

    assert len(nodes) >= 12
    # Ensure Annexure A and Points are detected
    annexure_nodes = [n for n in nodes if n.node_type == HierarchyNodeType.ANNEXURE]
    point_nodes = [n for n in nodes if n.node_type == HierarchyNodeType.POINT]

    assert len(annexure_nodes) >= 1
    assert len(point_nodes) >= 12


def test_hierarchical_chunker(sample_pdf_dir: Path) -> None:
    pdf_path = sample_pdf_dir / "circular_2022_50957.pdf"
    pages, _ = PDFParser.parse_pdf(pdf_path)

    builder = HierarchyBuilder()
    nodes = builder.build_hierarchy("doc-2022", pages)

    chunker = HierarchicalChunker(max_tokens=500)
    chunks = chunker.create_chunks("doc-2022", nodes)

    assert len(chunks) >= 12
    # Check sequential link wiring
    for i in range(len(chunks)):
        if i > 0:
            assert chunks[i].previous_chunk_id == chunks[i - 1].chunk_id
        if i < len(chunks) - 1:
            assert chunks[i].next_chunk_id == chunks[i + 1].chunk_id

        # Verify heading path is populated
        assert len(chunks[i].heading_path) >= 1


def test_hierarchy_builder_2026_amendment_with_faqs(sample_pdf_dir: Path) -> None:
    pdf_path = sample_pdf_dir / "circular_2026_amendment.pdf"
    pages, _ = PDFParser.parse_pdf(pdf_path)

    builder = HierarchyBuilder()
    nodes = builder.build_hierarchy("doc-2026", pages)

    faq_nodes = [n for n in nodes if n.node_type in [HierarchyNodeType.FAQ, HierarchyNodeType.QUESTION, HierarchyNodeType.ANSWER]]
    assert len(faq_nodes) >= 3
