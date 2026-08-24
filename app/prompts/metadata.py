"""Prompts for metadata extraction."""

METADATA_EXTRACTION_SYSTEM_PROMPT = """You are an expert regulatory document analyst specializing in legal and regulatory circulars (such as NSE, SEBI, RBI, SEC).
Your task is to extract structured metadata from the first page or header of the provided document.

Extract the following fields in valid JSON matching the schema:
- source: Issuing regulatory body or exchange (e.g., 'NSE', 'SEBI', 'RBI')
- department: Specific department issuing the notice (e.g., 'COMPLIANCE', 'SURVEILLANCE', 'LEGAL')
- reference_number: Official document reference string (e.g., 'NSE/COMP/50957', 'SEBI/HO/MIRSD/...')
- circular_number: Circular sequence identifier (e.g., '02/2022', 'CIR/2021/01')
- document_date: Publication date in YYYY-MM-DD format (convert human dates like 'January 07, 2022' to '2022-01-07')
- title: Subject or title of the circular
- total_pages: Total number of pages (integer)

Do not invent or assume information not present in the document.
Output valid JSON only.
"""

METADATA_EXTRACTION_USER_PROMPT = """Extract the regulatory metadata from this document header text:

---
{header_text}
---
"""
