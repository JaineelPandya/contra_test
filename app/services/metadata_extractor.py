"""Regex and LLM-fallback metadata extraction for regulatory circulars."""
from datetime import datetime
import re
from typing import Optional
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.models.document import DocumentMetadata
from app.services.pdf_parser import PDFPage


# Month name to number mapping for Indian/standard date formatting
MONTHS = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def parse_date_to_iso(date_str: str) -> Optional[str]:
    """Convert various date formats (e.g. 'January 07, 2022', '07-01-2022', '2022-01-07') to YYYY-MM-DD."""
    if not date_str:
        return None

    cleaned = date_str.strip()

    # Pattern: Month DD, YYYY or Month DD YYYY (e.g. January 07, 2022 or Jan 7 2022)
    m1 = re.search(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", cleaned)
    if m1:
        month_name = m1.group(1).lower()
        day = int(m1.group(2))
        year = int(m1.group(3))
        if month_name in MONTHS:
            return f"{year:04d}-{MONTHS[month_name]}-{day:02d}"

    # Pattern: DD Month YYYY (e.g. 07 January 2022, 07-Jan-2022, 07/Jan/2022)
    m2 = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[-/\s]+([A-Za-z]+),?[-/\s]+(\d{4})", cleaned)
    if m2:
        day = int(m2.group(1))
        month_name = m2.group(2).lower()
        year = int(m2.group(3))
        if month_name in MONTHS:
            return f"{year:04d}-{MONTHS[month_name]}-{day:02d}"

    # Pattern: YYYY-MM-DD
    m3 = re.search(r"(\d{4})-(\d{2})-(\d{2})", cleaned)
    if m3:
        return f"{m3.group(1)}-{m3.group(2)}-{m3.group(3)}"

    # Pattern: DD-MM-YYYY or DD/MM/YYYY
    m4 = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", cleaned)
    if m4:
        day = int(m4.group(1))
        month = int(m4.group(2))
        year = int(m4.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"

    return None


class MetadataExtractor:
    """Extracts metadata using deterministic regex patterns, falling back to LLM when needed."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self.llm_provider = llm_provider

    async def extract_metadata(self, pages: list[PDFPage], total_pages: int) -> DocumentMetadata:
        """Extract metadata from the first page header, using regex first and LLM as fallback."""
        header_text = pages[0].text if pages else ""
        meta = self._extract_regex(header_text, total_pages)

        # Check if critical metadata is missing
        missing_critical = not (meta.reference_number and meta.document_date and meta.title)

        if missing_critical:
            logger.info("Some metadata fields missing from regex extraction. Calling LLM fallback...")
            try:
                llm_meta = await self.llm_provider.extract_metadata(header_text[:2000])
                if llm_meta:
                    meta = self._merge_metadata(meta, llm_meta)
            except Exception as e:
                logger.warning("LLM metadata extraction failed, using regex results: %s", e)

        logger.info(
            "Extracted metadata: source=%s, ref=%s, cir_no=%s, date=%s, title=%s",
            meta.source,
            meta.reference_number,
            meta.circular_number,
            meta.document_date,
            meta.title,
        )
        return meta

    def _extract_regex(self, text: str, total_pages: int) -> DocumentMetadata:
        """Apply regex patterns to detect typical regulatory circular fields."""
        source = None
        department = None
        reference_number = None
        circular_number = None
        document_date = None
        title = None

        # Source detection
        if re.search(r"\b(National Stock Exchange of India|NSE)\b", text, re.IGNORECASE):
            source = "NSE"
        elif re.search(r"\b(Securities and Exchange Board of India|SEBI)\b", text, re.IGNORECASE):
            source = "SEBI"
        elif re.search(r"\b(Reserve Bank of India|RBI)\b", text, re.IGNORECASE):
            source = "RBI"
        elif re.search(r"\b(Bombay Stock Exchange|BSE)\b", text, re.IGNORECASE):
            source = "BSE"

        # Department detection
        dept_match = re.search(
            r"Department\s*:\s*([A-Z\s]+?)(?:\n|Circular|Date|Ref)",
            text,
            re.IGNORECASE,
        )
        if dept_match:
            department = dept_match.group(1).strip().upper()
        elif re.search(r"\b(COMPLIANCE|SURVEILLANCE|INSPECTION|LEGAL|MEMBERSHIP)\b", text, re.IGNORECASE):
            m = re.search(r"\b(COMPLIANCE|SURVEILLANCE|INSPECTION|LEGAL|MEMBERSHIP)\b", text, re.IGNORECASE)
            if m:
                department = m.group(1).upper()

        # Reference number: e.g. NSE/COMP/50957, SEBI/HO/MIRSD/...
        ref_match = re.search(r"((?:NSE|SEBI|RBI|BSE|CIR)[A-Za-z0-9_\-\/]+)", text)
        if ref_match:
            cand = ref_match.group(1).strip(" /-,.\n")
            if "/" in cand and len(cand) >= 6:
                reference_number = cand

        # Download / Circular Ref No: e.g. "Download Ref No: NSE/COMP/50957" or "Circular Ref. No: 02/2022"
        cir_ref_match = re.search(
            r"Circular\s*(?:Ref\.?\s*No|Number)?\s*[:\.]?\s*([0-9\/\-]+)",
            text,
            re.IGNORECASE,
        )
        if cir_ref_match:
            circular_number = cir_ref_match.group(1).strip()

        # Date detection
        date_match = re.search(
            r"Date\s*:\s*([A-Za-z0-9\s,\-\/]+?)(?:\n|To|All|Sub)",
            text,
            re.IGNORECASE,
        )
        if date_match:
            document_date = parse_date_to_iso(date_match.group(1))

        if not document_date:
            # Try searching for any standard date pattern in header
            date_cand = re.search(
                r"\b(?:dated\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/][A-Za-z0-9]+[-/]\d{4})\b",
                text,
                re.IGNORECASE,
            )
            if date_cand:
                document_date = parse_date_to_iso(date_cand.group(1))

        # Title / Subject detection
        subj_match = re.search(
            r"(?:Sub(?:ject)?|Title)\s*:\s*(.+?)(?:\n\n|\n[A-Z0-9\.\(\)\-]+\s+|Dear\s+Sir|Dear\s+Members)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if subj_match:
            title = re.sub(r"\s+", " ", subj_match.group(1)).strip()

        return DocumentMetadata(
            source=source,
            department=department,
            reference_number=reference_number,
            circular_number=circular_number,
            document_date=document_date,
            title=title,
            total_pages=total_pages,
        )

    def _merge_metadata(self, base: DocumentMetadata, fallback: DocumentMetadata) -> DocumentMetadata:
        """Merge regex-extracted metadata with LLM-extracted metadata."""
        return DocumentMetadata(
            source=base.source or fallback.source,
            department=base.department or fallback.department,
            reference_number=base.reference_number or fallback.reference_number,
            circular_number=base.circular_number or fallback.circular_number,
            document_date=base.document_date or parse_date_to_iso(fallback.document_date or "") or fallback.document_date,
            title=base.title or fallback.title,
            total_pages=base.total_pages,
        )
