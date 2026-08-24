"""Mock LLM Provider for unit testing and offline execution."""
import re
from typing import Any, Callable, Dict, List, Optional
from app.llm.base import BaseLLMProvider
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact, FactPredicate, FactStatus
from app.models.reference import DocumentReference


class MockProvider(BaseLLMProvider):
    """Rule-based mock provider for testing without an active LLM backend."""

    async def extract_metadata(self, header_text: str) -> Optional[DocumentMetadata]:
        return DocumentMetadata(
            source="NSE",
            department="COMPLIANCE",
            reference_number="NSE/COMP/50957",
            circular_number="02/2022",
            document_date="2022-01-07",
            title="Clarification to Rule 8(1)(f)",
            total_pages=2,
        )

    async def extract_facts_batch(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
        facts: List[Fact] = []
        for chunk in chunks:
            lines = [line.strip() for line in chunk.content.split("\n") if line.strip()]
            for line in lines:
                if len(line) < 10:
                    continue

                predicate = FactPredicate.REQUIRES
                regulatory_status = "ACTIVE"
                if re.search(r"\b(prohibit|shall not|must not|cannot|forbidden)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.PROHIBITS
                    regulatory_status = "PROHIBITED"
                elif re.search(r"\b(permit|allowed|may|authorized)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.PERMITS
                    regulatory_status = "PERMITTED"
                elif re.search(r"\b(clarif|guidance|explained)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.CLARIFIES
                elif re.search(r"\b(amend|modify|substitut)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.MODIFIES
                elif re.search(r"\b(delete|repeal|cancel)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.DELETES

                fact = Fact(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    hierarchy_node_id=chunk.parent_id,
                    subject="Trading Member",
                    predicate=predicate,
                    object=line[:100],
                    conditions=[],
                    exceptions=[],
                    regulatory_status=regulatory_status,
                    confidence=0.9,
                    source_text=line,
                    extraction_method="mock",
                    status=FactStatus.ACTIVE,
                    valid_from=metadata.document_date if metadata else None,
                )
                facts.append(fact)
        return facts

    async def classify_relationships_batch(
        self,
        candidate_pairs: List[Dict[str, Any]],
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for pair in candidate_pairs:
            src_text = pair.get("source_text", "").lower()
            rel = "NONE"
            confidence = 0.9

            if "amend" in src_text or "substitut" in src_text or "modifi" in src_text:
                rel = "MODIFIES"
            elif "delete" in src_text or "repeal" in src_text:
                rel = "DELETES"
            elif "clarif" in src_text or "faq" in src_text:
                rel = "CLARIFIES"
            elif "supersed" in src_text or "replac" in src_text:
                rel = "SUPERSEDES"
            elif "refer" in src_text or "vide" in src_text:
                rel = "REFERENCES"

            results.append({
                "source_id": pair.get("source_id"),
                "target_id": pair.get("target_id"),
                "relationship": rel,
                "confidence": confidence,
                "reason": f"Matched pattern in text: {rel}",
                "extraction_method": "mock",
            })
        return results

    async def detect_references(
        self,
        source_document_id: str,
        text: str,
    ) -> List[DocumentReference]:
        refs: List[DocumentReference] = []
        match = re.search(r"(NSE/[A-Z]+/\d+)", text)
        if match:
            refs.append(
                DocumentReference(
                    source_document_id=source_document_id,
                    target_reference_number=match.group(1),
                    reference_text=match.group(0),
                    reference_type="REFERENCES",
                )
            )
        return refs
