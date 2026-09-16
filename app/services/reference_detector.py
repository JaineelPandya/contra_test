"""Cross-document and cross-clause reference detection service."""
import re
from typing import List, Optional
from app.core.logging import logger
from app.graph.repository import GraphRepository
from app.llm.base import BaseLLMProvider
from app.models.chunk import Chunk
from app.models.reference import DocumentReference
from app.services.metadata_extractor import parse_date_to_iso


def normalize_reference_number(ref: Optional[str]) -> Optional[str]:
    """Normalize a circular/document reference number to canonical form.

    Examples:
      'NSE / COMP / 50957' -> 'NSE/COMP/50957'
      'SEBI/HO/OIAE/OIAE_IAD-1/P/CIR/2023/131' -> 'SEBI/HO/OIAE/OIAE_IAD-1/P/CIR/2023/131'
    """
    if not ref:
        return None
    cleaned = ref.strip().upper()
    # Remove whitespace around slashes and dashes
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    # Standardize NSE/<DEPT>/<NUM>
    m = re.match(r"^NSE(?:/|-)?([A-Z]{2,6})(?:/|-)?([0-9]{4,6})$", cleaned)
    if m:
        return f"NSE/{m.group(1)}/{m.group(2)}"
    return cleaned


class ReferenceDetector:
    """Detects citations in new documents pointing to existing documents in Neo4j."""

    def __init__(self, repository: GraphRepository, llm_provider: BaseLLMProvider) -> None:
        self.repository = repository
        self.llm_provider = llm_provider

    async def detect_references(self, document_id: str, chunks: List[Chunk]) -> List[DocumentReference]:
        """Scan document chunks for explicit and implicit citations to prior circulars."""
        logger.info("Detecting cross-document references for document %s (scanning %d chunks)", document_id, len(chunks))
        # Scan full document text across all chunks to ensure citations in preamble or body are not missed
        combined_text = "\n".join([c.content for c in chunks])

        detected_references: List[DocumentReference] = []

        # 1. Deterministic regex detection
        regex_refs = self._detect_regex_references(document_id, combined_text)
        detected_references.extend(regex_refs)

        # 2. LLM fallback detection if regex didn't find anything
        if not detected_references:
            try:
                preamble_text = "\n".join([c.content for c in chunks[:5]])
                llm_refs = await self.llm_provider.detect_references(document_id, preamble_text)
                detected_references.extend(llm_refs)
            except Exception as e:
                logger.warning("LLM reference detection failed: %s", e)

        # 3. Deduplicate references by target_reference_number / target_circular_number
        unique_refs: List[DocumentReference] = []
        seen_targets = set()
        for r in detected_references:
            norm_target = normalize_reference_number(r.target_reference_number) or r.target_circular_number
            if norm_target and norm_target not in seen_targets:
                seen_targets.add(norm_target)
                r.target_reference_number = normalize_reference_number(r.target_reference_number)
                unique_refs.append(r)

        # 4. Resolve target_document_id by querying Neo4j Document.reference_number
        resolved_references: List[DocumentReference] = []
        for ref in unique_refs:
            norm_ref = normalize_reference_number(ref.target_reference_number)
            logger.info(
                "Extracted reference | raw='%s' | normalized='%s' | circular_num='%s' | date='%s' | snippet='%s'",
                ref.target_reference_number,
                norm_ref,
                ref.target_circular_number,
                ref.target_date,
                ref.reference_text[:120].replace("\n", " "),
            )

            target_doc = self.repository.find_document_by_reference(
                reference_number=norm_ref,
                circular_number=ref.target_circular_number,
                date=ref.target_date,
                exclude_document_id=document_id,
            )

            if target_doc and target_doc.document_id != document_id:
                ref.target_document_id = target_doc.document_id
                logger.info(
                    "Target document FOUND in Neo4j | target_document_id=%s | reference_number=%s | filename=%s",
                    target_doc.document_id,
                    target_doc.reference_number,
                    target_doc.filename,
                )
                # Persist cross-document relationship (source:Document)-[:REFERENCES_DOC]->(target:Document)
                self.repository.save_reference(ref)
                logger.info(
                    "Persisted cross-document relationship: (Document {id=%s})-[:REFERENCES_DOC]->(Document {id=%s, ref=%s})",
                    document_id,
                    target_doc.document_id,
                    target_doc.reference_number,
                )
            else:
                logger.warning(
                    "Target document NOT found in Neo4j | lookup_reference='%s' | circular_number='%s' | date='%s'",
                    norm_ref,
                    ref.target_circular_number,
                    ref.target_date,
                )

            resolved_references.append(ref)

        logger.info(
            "Reference detection complete for document %s: found %d references (%d resolved to existing documents)",
            document_id,
            len(resolved_references),
            sum(1 for r in resolved_references if r.target_document_id),
        )
        return resolved_references

    def _detect_ref_type(self, text_context: str) -> str:
        """Classify relationship type from surrounding regulatory formula context."""
        ctx = text_context.lower()
        if re.search(r"\b(supersed(?:es|ed|ing)?|in\s+supersession\s+of|replaces?|repeals?)\b", ctx):
            return "SUPERSEDES"
        if re.search(r"\b(clarif(?:ies|ication|ied)|is\s+clarified)\b", ctx):
            return "CLARIFIES"
        if re.search(r"\b(partial\s+modification|modifi(?:es|ed|cation))\b", ctx):
            return "MODIFIES"
        if re.search(r"\b(amend(?:s|ed|ment)?|substitut(?:es|ed)|stands?\s+deleted|are\s+deleted|revis(?:es|ed))\b", ctx):
            return "AMENDS"
        return "REFERENCES"

    def _detect_regex_references(self, source_doc_id: str, text: str) -> List[DocumentReference]:
        """Detect standard circular reference patterns using robust regular expressions."""
        refs: List[DocumentReference] = []
        seen_refs = set()

        doc_level_type = self._detect_ref_type(text[:1500])

        # 1. Match explicit standard NSE references: NSE/COMP/50957, NSE / COMP / 50957, NSE/COMP50957, NSE/INSP/74836
        nse_matches = re.finditer(
            r"\b(NSE\s*/?\s*[A-Za-z]{2,6}\s*/?\s*[0-9]{4,6})\b",
            text,
            re.IGNORECASE,
        )
        for m in nse_matches:
            raw_match = m.group(1).strip()
            norm_match = normalize_reference_number(raw_match)
            if norm_match and norm_match not in seen_refs:
                seen_refs.add(norm_match)

                # Look around match for associated date and local context (e.g. "dated January 07, 2022")
                start_pos = max(0, m.start() - 100)
                end_pos = min(len(text), m.end() + 150)
                window = text[start_pos:end_pos]

                date_match = re.search(
                    r"dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
                    window,
                    re.IGNORECASE,
                )
                iso_date = parse_date_to_iso(date_match.group(1)) if date_match else None
                local_type = self._detect_ref_type(window)
                ref_type = local_type if local_type != "REFERENCES" else doc_level_type

                ref = DocumentReference(
                    source_document_id=source_doc_id,
                    target_reference_number=norm_match,
                    target_date=iso_date,
                    reference_text=window.strip(),
                    reference_type=ref_type,
                    confidence=1.0,
                )
                refs.append(ref)

        # 2. Match SEBI/CIR/RBI/BSE references: e.g. SEBI/HO/OIAE/OIAE_IAD-1/P/CIR/2023/131, CIR/MIRSD/11/2013
        sebi_matches = re.finditer(
            r"\b((?:SEBI|RBI|BSE|CIR)\s*/\s*[A-Za-z0-9_\-\/]+/[0-9]{1,5})\b",
            text,
            re.IGNORECASE,
        )
        for m in sebi_matches:
            raw_match = m.group(1).strip()
            norm_match = normalize_reference_number(raw_match)
            if norm_match and norm_match not in seen_refs and not norm_match.startswith("CIRCULAR/"):
                seen_refs.add(norm_match)
                start_pos = max(0, m.start() - 100)
                end_pos = min(len(text), m.end() + 150)
                window = text[start_pos:end_pos]
                date_match = re.search(
                    r"dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
                    window,
                    re.IGNORECASE,
                )
                iso_date = parse_date_to_iso(date_match.group(1)) if date_match else None
                local_type = self._detect_ref_type(window)
                ref_type = local_type if local_type != "REFERENCES" else doc_level_type
                ref = DocumentReference(
                    source_document_id=source_doc_id,
                    target_reference_number=norm_match,
                    target_date=iso_date,
                    reference_text=window.strip(),
                    reference_type=ref_type,
                    confidence=1.0,
                )
                refs.append(ref)

        # 3. Match generic circular reference syntax: "Circular Ref No: 02/2022", "Circular No. 15/2026"
        circ_matches = re.finditer(
            r"(?:circular|ref\.?\s*no\.?|reference\s+to|vide\s+circular)\s+(?:no\.?\s*)?([0-9]{1,4}\s*/\s*20[0-9]{2})\b(?:\s+dated\s+([A-Za-z0-9\s,]+?))?",
            text,
            re.IGNORECASE,
        )
        for m in circ_matches:
            circ_num = m.group(1).replace(" ", "").strip()
            date_str = m.group(2).strip(" .,-") if len(m.groups()) > 1 and m.group(2) else None
            iso_date = parse_date_to_iso(date_str) if date_str else None

            if circ_num and circ_num not in seen_refs:
                seen_refs.add(circ_num)
                start_pos = max(0, m.start() - 100)
                end_pos = min(len(text), m.end() + 150)
                window = text[start_pos:end_pos]
                local_type = self._detect_ref_type(window)
                ref_type = local_type if local_type != "REFERENCES" else doc_level_type

                ref = DocumentReference(
                    source_document_id=source_doc_id,
                    target_circular_number=circ_num,
                    target_date=iso_date,
                    reference_text=window.strip(),
                    reference_type=ref_type,
                    confidence=1.0,
                )
                refs.append(ref)

        # 4. Match Date-based circular citations: "Circular dated July 31, 2023"
        date_ref_matches = re.finditer(
            r"(?:circular|amendment|corrigendum)\s+(?:dated|of)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
            text,
            re.IGNORECASE,
        )
        for m in date_ref_matches:
            raw_date = m.group(1).strip()
            iso_date = parse_date_to_iso(raw_date)
            date_key = f"date_{iso_date}"
            if iso_date and date_key not in seen_refs:
                seen_refs.add(date_key)
                start_pos = max(0, m.start() - 100)
                end_pos = min(len(text), m.end() + 150)
                window = text[start_pos:end_pos]
                local_type = self._detect_ref_type(window)
                ref_type = local_type if local_type != "REFERENCES" else doc_level_type
                ref = DocumentReference(
                    source_document_id=source_doc_id,
                    target_date=iso_date,
                    reference_text=window.strip(),
                    reference_type=ref_type,
                    confidence=0.9,
                )
                refs.append(ref)

        return refs

