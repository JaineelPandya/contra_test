"""Unit tests for PyMuPDF parser and metadata extractor."""
from pathlib import Path
import pytest
from app.llm.mock_provider import MockProvider
from app.services.metadata_extractor import MetadataExtractor, parse_date_to_iso
from app.services.pdf_parser import PDFParser


def test_parse_date_to_iso() -> None:
    assert parse_date_to_iso("January 07, 2022") == "2022-01-07"
    assert parse_date_to_iso("07-Jan-2022") == "2022-01-07"
    assert parse_date_to_iso("2026-06-23") == "2026-06-23"
    assert parse_date_to_iso("23 June 2026") == "2026-06-23"


def test_pdf_parser_on_synthetic_circulars(sample_pdf_dir: Path) -> None:
    pdf_path = sample_pdf_dir / "circular_2022_50957.pdf"
    pages, total_pages = PDFParser.parse_pdf(pdf_path)

    assert total_pages == 2
    assert len(pages) == 2
    assert "NSE/COMP/50957" in pages[0].text
    assert "Point No. 1" in pages[0].text
    assert "Point No. 7" in pages[1].text


@pytest.mark.asyncio
async def test_metadata_extractor(sample_pdf_dir: Path, mock_llm_provider: MockProvider) -> None:
    pdf_path = sample_pdf_dir / "circular_2022_50957.pdf"
    pages, total_pages = PDFParser.parse_pdf(pdf_path)

    extractor = MetadataExtractor(mock_llm_provider)
    meta = await extractor.extract_metadata(pages, total_pages)

    assert meta.source == "NSE"
    assert meta.department == "COMPLIANCE"
    assert meta.reference_number == "NSE/COMP/50957"
    assert meta.circular_number == "02/2022"
    assert meta.document_date == "2022-01-07"
    assert "Rule 8(1)(f)" in (meta.title or "")
