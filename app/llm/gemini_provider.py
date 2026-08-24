"""Google Gemini LLM Provider implementation with batching and request manager."""
import json
import re
from typing import Any, Callable, Dict, List, Optional
from app.core.config import get_settings
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.llm.request_manager import gemini_request_manager
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact, FactPredicate, FactStatus
from app.models.reference import DocumentReference
from app.prompts.metadata import METADATA_EXTRACTION_SYSTEM_PROMPT, METADATA_EXTRACTION_USER_PROMPT
from app.prompts.fact_extraction import BATCH_FACT_EXTRACTION_SYSTEM_PROMPT, BATCH_FACT_EXTRACTION_USER_PROMPT
from app.prompts.change_detection import BATCH_RELATIONSHIP_SYSTEM_PROMPT, BATCH_RELATIONSHIP_USER_PROMPT
from app.prompts.reference_detection import REFERENCE_DETECTION_SYSTEM_PROMPT, REFERENCE_DETECTION_USER_PROMPT


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Strip markdown code fences and clean JSON output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    return json.loads(cleaned)


class GeminiProvider(BaseLLMProvider):
    """LLM Provider interfacing directly with Google Gemini REST API via GeminiRequestManager."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        self.model = self.settings.GEMINI_MODEL
        self.temperature = self.settings.GEMINI_TEMPERATURE
        self.timeout = 60.0
        self.request_manager = gemini_request_manager

    async def _generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        on_dead_letter: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Send generateContent request through GeminiRequestManager enforcing JSON output."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": self.temperature,
            },
        }

        data = await self.request_manager.execute_request(
            url=url,
            payload=payload,
            headers=headers,
            timeout=self.timeout,
            on_dead_letter=on_dead_letter,
        )
        text_content = data["candidates"][0]["content"]["parts"][0]["text"]
        return clean_json_response(text_content)

    async def extract_metadata(self, header_text: str) -> Optional[DocumentMetadata]:
        """Extract structured document metadata using Gemini."""
        try:
            user_prompt = METADATA_EXTRACTION_USER_PROMPT.format(header_text=header_text)
            data = await self._generate_json(METADATA_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            return DocumentMetadata(**data)
        except Exception as e:
            logger.warning("Gemini metadata extraction failed: %s", e)
            return None

    async def extract_facts_batch(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
        """Extract atomic regulatory facts from a batch of chunks in a single call."""
        if not chunks:
            return []

        chunks_payload = [
            {
                "chunk_id": c.chunk_id,
                "heading_path": " > ".join(c.heading_path) if c.heading_path else "Section",
                "text": c.content,
            }
            for c in chunks
        ]

        def dead_letter_callback(status_code: int, error_msg: str) -> None:
            if on_dead_letter:
                on_dead_letter(status_code, error_msg, {"chunks": chunks_payload})

        try:
            meta_json = metadata.model_dump_json() if metadata else "{}"
            user_prompt = BATCH_FACT_EXTRACTION_USER_PROMPT.format(
                metadata_json=meta_json,
                chunks_json=json.dumps(chunks_payload, indent=2),
            )

            data = await self._generate_json(
                BATCH_FACT_EXTRACTION_SYSTEM_PROMPT,
                user_prompt,
                on_dead_letter=dead_letter_callback,
            )

            results = data.get("results", [])
            facts: List[Fact] = []
            chunk_map = {c.chunk_id: c for c in chunks}

            for res in results:
                cid = res.get("chunk_id")
                matched_chunk = chunk_map.get(cid)
                raw_facts = res.get("facts", [])

                for rf in raw_facts:
                    pred_str = str(rf.get("predicate", "REQUIRES")).upper()
                    try:
                        predicate = FactPredicate(pred_str)
                    except ValueError:
                        predicate = FactPredicate.REQUIRES

                    fact = Fact(
                        document_id=document_id,
                        chunk_id=cid if cid else chunks[0].chunk_id,
                        hierarchy_node_id=matched_chunk.parent_id if matched_chunk else None,
                        subject=rf.get("subject", "Regulated Entity"),
                        predicate=predicate,
                        object=rf.get("object", ""),
                        conditions=rf.get("conditions", []),
                        exceptions=rf.get("exceptions", []),
                        regulatory_status=rf.get("regulatory_status", "ACTIVE"),
                        effective_date=rf.get("effective_date"),
                        confidence=float(rf.get("confidence", 0.95)),
                        source_text=rf.get("source_text", matched_chunk.content[:200] if matched_chunk else ""),
                        extraction_method="gemini",
                        model=self.model,
                        status=FactStatus.ACTIVE,
                        valid_from=metadata.document_date if metadata else None,
                    )
                    facts.append(fact)

            logger.info("Successfully extracted %d facts via Gemini for batch of %d chunks", len(facts), len(chunks))
            return facts

        except Exception as e:
            logger.warning("Gemini batched fact extraction failed (%s). Executing rule-based fallback.", e)
            return self._fallback_fact_extraction(document_id, chunks, metadata)

    async def classify_relationships_batch(
        self,
        candidate_pairs: List[Dict[str, Any]],
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of candidate relationship pairs with controlled vocabulary."""
        if not candidate_pairs:
            return []

        def dead_letter_callback(status_code: int, error_msg: str) -> None:
            if on_dead_letter:
                on_dead_letter(status_code, error_msg, {"candidate_pairs": candidate_pairs})

        try:
            pairs_payload = [
                {
                    "source_id": p.get("source_id"),
                    "target_id": p.get("target_id"),
                    "source_text": p.get("source_text", ""),
                    "target_text": p.get("target_text", ""),
                }
                for p in candidate_pairs
            ]

            user_prompt = BATCH_RELATIONSHIP_USER_PROMPT.format(
                candidate_pairs_json=json.dumps(pairs_payload, indent=2)
            )

            data = await self._generate_json(
                BATCH_RELATIONSHIP_SYSTEM_PROMPT,
                user_prompt,
                on_dead_letter=dead_letter_callback,
            )

            results = data.get("results", [])
            for r in results:
                r["extraction_method"] = "gemini"
                r["model"] = self.model

            return results

        except Exception as e:
            logger.warning("Gemini batched relationship classification failed (%s).", e)
            return []

    async def detect_references(
        self,
        source_document_id: str,
        text: str,
    ) -> List[DocumentReference]:
        """Detect citations and cross-document references from text."""
        try:
            user_prompt = REFERENCE_DETECTION_USER_PROMPT.format(text=text)
            data = await self._generate_json(REFERENCE_DETECTION_SYSTEM_PROMPT, user_prompt)
            raw_refs = data.get("references", [])
            refs: List[DocumentReference] = []

            for rr in raw_refs:
                ref = DocumentReference(
                    source_document_id=source_document_id,
                    target_reference_number=rr.get("target_reference_number"),
                    target_circular_number=rr.get("target_circular_number"),
                    target_date=rr.get("target_date"),
                    target_point_or_clause=rr.get("target_point_or_clause"),
                    reference_text=rr.get("reference_text", text[:100]),
                    reference_type=rr.get("reference_type", "REFERENCES"),
                    confidence=float(rr.get("confidence", 0.95)),
                )
                refs.append(ref)
            return refs
        except Exception as e:
            logger.warning("Gemini reference detection failed: %s", e)
            return []

    # ==================== DETERMINISTIC FALLBACK EXTRACTION ====================

    def _fallback_fact_extraction(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
    ) -> List[Fact]:
        """Deterministic rule-based fallback when Gemini API calls are exhausted."""
        facts: List[Fact] = []

        for chunk in chunks:
            lines = [line.strip() for line in chunk.content.split("\n") if line.strip()]
            for line in lines:
                if len(line) < 15:
                    continue

                predicate = FactPredicate.REQUIRES
                regulatory_status = "ACTIVE"

                if re.search(r"\b(prohibit|shall not|must not|cannot|forbidden|barred)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.PROHIBITS
                    regulatory_status = "PROHIBITED"
                elif re.search(r"\b(permit|allowed|may|authorized|eligible)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.PERMITS
                    regulatory_status = "PERMITTED"
                elif re.search(r"\b(clarif|guidance|faq|explained)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.CLARIFIES
                elif re.search(r"\b(amend|modify|substitut|revised)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.MODIFIES
                elif re.search(r"\b(delete|repeal|cancel|withdrawn)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.DELETES
                    regulatory_status = "DELETED"
                elif re.search(r"\b(define|means|shall be understood)\b", line, re.IGNORECASE):
                    predicate = FactPredicate.DEFINES

                subject = "Trading Member"
                if re.search(r"\b(Exchange|NSE|SEBI|Board)\b", line, re.IGNORECASE):
                    subject = "Exchange / Regulator"
                elif re.search(r"\b(Client|Investor)\b", line, re.IGNORECASE):
                    subject = "Client / Investor"

                fact = Fact(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    hierarchy_node_id=chunk.parent_id,
                    subject=subject,
                    predicate=predicate,
                    object=line[:120],
                    conditions=[],
                    exceptions=[],
                    regulatory_status=regulatory_status,
                    confidence=0.60,
                    source_text=line,
                    extraction_method="rule_based",
                    model=None,
                    fallback_reason="gemini_retry_exhausted",
                    status=FactStatus.ACTIVE,
                    valid_from=metadata.document_date if metadata else None,
                )
                facts.append(fact)

        return facts
