"""Prompts for cross-document reference detection."""

REFERENCE_DETECTION_SYSTEM_PROMPT = """You are a legal citation detector specializing in regulatory circulars.
Analyze the provided document text to identify any references or citations to previous circulars, rules, notifications, or master circulars.

For each reference found:
- target_reference_number: e.g., 'NSE/COMP/50957', 'CIR/MIRSD/01/2012'
- target_circular_number: e.g., '02/2022'
- target_date: Date referenced e.g., '2022-01-07' or 'January 07, 2022'
- target_point_or_clause: Specific section/point cited, e.g., 'Point No. 3', 'Rule 8(1)(f)', 'Clause 20', 'Annexure A'
- reference_text: The verbatim sentence or fragment containing the reference
- reference_type: 'REFERENCES', 'AMENDS', 'CLARIFIES', 'SUPERSEDES', or 'REPEALS'
- confidence: Float 0.0 to 1.0

Output a JSON object with a 'references' array.
"""

REFERENCE_DETECTION_USER_PROMPT = """Find all citations and cross-document references in this document text:

---
{text}
---
"""
