"""Ollama LLM Provider implementation with batching support."""
import json
import re
from typing import Any, Callable, Dict, List, Optional
import httpx
from app.core.config import get_settings
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
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


class OllamaProvider(BaseLLMProvider):
    """LLM Provider interfacing with local or remote Ollama server."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = self.settings.OLLAMA_MODEL
        self.temperature = self.settings.OLLAMA_TEMPERATURE
        self.timeout = 120.0

    async def _generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Send chat request to Ollama and enforce JSON response."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "{}")
            return clean_json_response(content)

    async def extract_metadata(self, header_text: str) -> Optional[DocumentMetadata]:
        try:
            user_prompt = METADATA_EXTRACTION_USER_PROMPT.format(header_text=header_text)
            data = await self._generate_json(METADATA_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            return DocumentMetadata(**data)
        except Exception as e:
            logger.warning("Ollama metadata extraction failed: %s", e)
            return None

    async def extract_facts_batch(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
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

        try:
            meta_json = metadata.model_dump_json() if metadata else "{}"
            user_prompt = BATCH_FACT_EXTRACTION_USER_PROMPT.format(
                metadata_json=meta_json,
                chunks_json=json.dumps(chunks_payload, indent=2),
            )

            data = await self._generate_json(BATCH_FACT_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            results = data.get("results", [])
            facts: List[Fact] = []
            chunk_map = {c.chunk_id: c for c in chunks}

            for res in results:
                cid = res.get("chunk_id")
                matched_chunk = chunk_map.get(cid)
                raw_facts = res.get("facts", [])

                for rf in raw_facts:
                    pred_str = str(rf.get("predicate", "")).strip().upper()
                    try:
                        predicate = FactPredicate(pred_str)
                    except ValueError:
                        logger.warning("Rejected fact: invalid_predicate chunk_id=%s predicate=%r", cid, pred_str)
                        continue

                    obj_str = str(rf.get("object", "")).strip()
                    if not obj_str:
                        logger.warning("Rejected fact: empty object in chunk_id=%s", cid)
                        continue

                    raw_cond = rf.get("condition")
                    conds = [raw_cond] if isinstance(raw_cond, str) and raw_cond.strip() else list(rf.get("conditions") or [])

                    raw_exc = rf.get("exception")
                    excs = [raw_exc] if isinstance(raw_exc, str) and raw_exc.strip() else list(rf.get("exceptions") or [])

                    fact = Fact(
                        document_id=document_id,
                        chunk_id=cid if cid else chunks[0].chunk_id,
                        hierarchy_node_id=matched_chunk.parent_id if matched_chunk else None,
                        subject=rf.get("subject", "Regulated Entity"),
                        predicate=predicate,
                        object=obj_str,
                        rule_reference=rf.get("rule_reference"),
                        clause_reference=rf.get("clause_reference"),
                        conditions=conds,
                        exceptions=excs,
                        threshold=rf.get("threshold"),
                        time=rf.get("time"),
                        source_circular=rf.get("source_circular"),
                        regulatory_status=rf.get("regulatory_status", "ACTIVE"),
                        effective_date=rf.get("effective_date"),
                        confidence=float(rf.get("confidence", 0.9)),
                        source_text=rf.get("source_text", matched_chunk.content[:200] if matched_chunk else ""),
                        extraction_method="ollama",
                        model=self.model,
                        status=FactStatus.ACTIVE,
                        valid_from=metadata.document_date if metadata else None,
                    )
                    facts.append(fact)

            return facts
        except Exception as e:
            logger.error("Ollama batched fact extraction failed: %s", e)
            if on_dead_letter:
                on_dead_letter(500, str(e), {"chunks": chunks_payload})
            return []

    async def classify_relationships_batch(
        self,
        candidate_pairs: List[Dict[str, Any]],
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidate_pairs:
            return []

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
            data = await self._generate_json(BATCH_RELATIONSHIP_SYSTEM_PROMPT, user_prompt)
            results = data.get("results", [])
            for r in results:
                r["extraction_method"] = "ollama"
                r["model"] = self.model
            return results
        except Exception as e:
            logger.error("Ollama relationship classification failed: %s", e)
            if on_dead_letter:
                on_dead_letter(500, str(e), {"candidate_pairs": candidate_pairs})
            return []

    async def detect_references(
        self,
        source_document_id: str,
        text: str,
    ) -> List[DocumentReference]:
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
                    confidence=float(rr.get("confidence", 0.9)),
                )
                refs.append(ref)
            return refs
        except Exception as e:
            logger.error("Ollama reference detection failed: %s", e)
            return []
