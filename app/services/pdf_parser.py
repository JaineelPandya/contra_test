"""PDF parsing service using PyMuPDF (fitz)."""
from pathlib import Path
from typing import Any, Dict, List, Tuple
import fitz  # PyMuPDF
from app.core.logging import logger


class PDFPage:
    """Represents a single parsed PDF page."""

    def __init__(self, page_number: int, text: str, blocks: List[Dict[str, Any]]) -> None:
        self.page_number = page_number
        self.text = text
        self.blocks = blocks


class PDFParser:
    """Extracts raw text, layout blocks, and page data from PDF files using PyMuPDF."""

    @staticmethod
    def parse_pdf(file_path: Path) -> Tuple[List[PDFPage], int]:
        """Open and extract all text and layout blocks from a PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at {file_path}")

        logger.info("Parsing PDF file: %s", file_path)
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages: List[PDFPage] = []

        try:
            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                page_text = page.get_text("text")
                # Extract text blocks with coordinates and layout metadata
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])
                pages.append(PDFPage(page_number=page_idx + 1, text=page_text, blocks=blocks))
        finally:
            doc.close()

        logger.info("Successfully extracted %d pages from %s", total_pages, file_path)
        return pages, total_pages

    @staticmethod
    def get_full_text(pages: List[PDFPage]) -> str:
        """Combine text from all pages with page markers."""
        full_text = []
        for page in pages:
            full_text.append(f"--- [Page {page.page_number}] ---\n{page.text}")
        return "\n\n".join(full_text)
