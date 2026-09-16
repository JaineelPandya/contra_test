"""OpenAI-compatible LLM Provider implementation with batching support."""
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


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM Provider interfacing with OpenAI or OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.OPENAI_BASE_URL.rstrip("/")
        self.api_key = self.settings.OPENAI_API_KEY
        self.model = self.settings.OPENAI_MODEL
        self.temperature = self.settings.OPENAI_TEMPERATURE
        self.timeout = 120.0

    async def _generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Send chat completion request to OpenAI API."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return clean_json_response(content)

    async def extract_metadata(self, header_text: str) -> Optional[DocumentMetadata]:
        try:
            user_prompt = METADATA_EXTRACTION_USER_PROMPT.format(header_text=header_text)
            data = await self._generate_json(METADATA_EXTRACTION_SYSTEM_PROMPT, user_prompt)
            return DocumentMetadata(**data)
        except Exception as e:
            logger.warning("OpenAI metadata extraction failed: %s", e)
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
                        extraction_method="openai",
                        model=self.model,
                        status=FactStatus.ACTIVE,
                        valid_from=metadata.document_date if metadata else None,
                    )
                    facts.append(fact)

            return facts
        except Exception as e:
            logger.error("OpenAI batched fact extraction failed: %s", e)
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
                r["extraction_method"] = "openai"
                r["model"] = self.model
            return results
        except Exception as e:
            logger.error("OpenAI relationship classification failed: %s", e)
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
                    confidence=float(rr.get("confidence", 0.95)),
                )
                refs.append(ref)
            return refs
        except Exception as e:
            logger.error("OpenAI reference detection failed: %s", e)
            return []
