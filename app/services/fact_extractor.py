"""Fact extraction service coordinating batched LLM fact extraction with token-aware batching."""
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.config import get_settings
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact
from app.prompts.fact_extraction import BATCH_FACT_EXTRACTION_SYSTEM_PROMPT
from app.services.chunker import approximate_token_count


def is_boilerplate_chunk(chunk: Chunk) -> Tuple[bool, str]:
    """Determine if a chunk contains only non-regulatory boilerplate.

    Filters out pure document headers, address salutations, signature blocks,
    and page footers to prevent wasting LLM token budget.
    """
    text = chunk.content.strip()
    if not text or len(text) < 25:
        return True, "Too short / empty content"

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Check signature blocks
    if re.search(
        r"^(?:Yours\s+faithfully|For\s+(?:National\s+Stock\s+Exchange|NSE|SEBI|RBI)|Authorized\s+Signatory|Chief\s+Manager|Senior\s+Manager)",
        text,
        re.IGNORECASE,
    ) and len(lines) <= 5:
        return True, "Signature block"

    # Check pure address / salutation blocks
    if re.search(
        r"^(?:To\s+All\s+(?:Members|Trading\s+Members|Clearing\s+Members)|Dear\s+Sir\s*/\s*Madam)",
        text,
        re.IGNORECASE,
    ) and len(lines) <= 4 and not re.search(r"\b(require|prohibit|permit|shall|must|amend|modify|delete|clarif|rule|point)\b", text, re.IGNORECASE):
        return True, "Address / Salutation block"

    # Check pure header letterhead blocks
    if re.search(
        r"^(?:NATIONAL\s+STOCK\s+EXCHANGE|CIRCULAR\s+Ref|Download\s+Ref\s+No|Date:\s+[A-Za-z]+)",
        text,
        re.IGNORECASE,
    ) and len(lines) <= 5 and not re.search(r"\b(require|prohibit|permit|shall|must|amend|modify|delete|clarif|rule|activity|point)\b", text, re.IGNORECASE):
        return True, "Header / Reference block without regulatory requirements"

    # Check footer / contact info blocks
    if re.search(
        r"^(?:Page\s+\d+\s+of\s+\d+|Regd\.?\s*Office\s*:|CIN\s*:|Tel\s*No|www\.nseindia\.com)",
        text,
        re.IGNORECASE,
    ) and len(lines) <= 3:
        return True, "Footer / Contact disclaimer block"

    return False, ""


from app.services.qualifier_extractor import extract_qualifier_spans



def validate_fact_evidence(fact: Fact, chunk: Chunk) -> Tuple[bool, Optional[str]]:
    """Strictly validate that a fact's evidence is grounded in the source chunk.

    Rules:
      - If evidence is empty -> reject (empty_evidence)
      - If normalized evidence is not in normalized chunk content -> reject (evidence_not_found)
      - Never fabricates evidence or injects chunk slices
    """
    evidence = (fact.source_text or "").strip()
    if not evidence:
        return False, "empty_evidence"

    from app.models.fact import normalize_text
    if normalize_text(evidence) not in normalize_text(chunk.content):
        return False, "evidence_not_found"

    return True, None


class FactExtractor:
    """Extracts atomic regulatory facts using token-aware batched LLM processing."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self.llm_provider = llm_provider
        self.settings = get_settings()
        self.max_input_tokens = self.settings.GROQ_MAX_REQUEST_INPUT_TOKENS
        self.max_chunks_per_batch = self.settings.FACT_EXTRACTION_BATCH_SIZE

    def create_token_aware_batches(
        self,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
    ) -> List[List[Chunk]]:
        """Group chunks into batches constrained by estimated token count and max input budget.

        Calculates system prompt overhead, metadata payload, and per-chunk token count
        (including JSON structure overhead) to ensure no single request exceeds the budget.
        """
        if not chunks:
            return []

        # Calculate baseline prompt overhead
        system_tokens = approximate_token_count(BATCH_FACT_EXTRACTION_SYSTEM_PROMPT)
        meta_json = metadata.model_dump_json() if metadata else "{}"
        meta_tokens = approximate_token_count(meta_json)
        template_overhead = 100  # JSON wrapper and prompt instructions
        base_prompt_tokens = system_tokens + meta_tokens + template_overhead

        batches: List[List[Chunk]] = []
        current_batch: List[Chunk] = []
        current_batch_tokens = base_prompt_tokens

        for chunk in chunks:
            chunk_content_tokens = chunk.token_count if chunk.token_count > 0 else approximate_token_count(chunk.content)
            # Add ~40 tokens for chunk JSON structure ("chunk_id", "heading_path", formatting)
            chunk_total_tokens = chunk_content_tokens + 40

            # If adding this chunk exceeds max input token budget and current batch is non-empty,
            # seal current batch and start a new one.
            exceeds_token_budget = (current_batch_tokens + chunk_total_tokens) > self.max_input_tokens
            exceeds_chunk_cap = bool(self.max_chunks_per_batch > 0 and len(current_batch) >= self.max_chunks_per_batch)

            if current_batch and (exceeds_token_budget or exceeds_chunk_cap):
                batches.append(current_batch)
                current_batch = [chunk]
                current_batch_tokens = base_prompt_tokens + chunk_total_tokens
            else:
                current_batch.append(chunk)
                current_batch_tokens += chunk_total_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def extract_all_facts(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
        """Extract atomic facts across all chunks grouped into token-aware batches."""
        all_facts: List[Fact] = []
        if not chunks:
            return all_facts

        # 1. Filter out non-regulatory boilerplate chunks
        relevant_chunks: List[Chunk] = []
        for c in chunks:
            is_bp, reason = is_boilerplate_chunk(c)
            if is_bp:
                logger.info(
                    "[BOILERPLATE_FILTER] Skipped non-regulatory chunk %s (Heading: '%s', Reason: %s)",
                    c.chunk_id,
                    c.path or c.heading_path,
                    reason,
                )
            else:
                relevant_chunks.append(c)

        if not relevant_chunks:
            logger.warning("No regulatory chunks remaining after boilerplate filter for doc %s", document_id)
            return all_facts

        # 2. Dynamically create token-aware chunk batches
        batches = self.create_token_aware_batches(relevant_chunks, metadata)

        logger.info(
            "Extracting facts for document %s (%d/%d relevant chunks packed into %d token-aware batches, budget=%d tokens)",
            document_id,
            len(relevant_chunks),
            len(chunks),
            len(batches),
            self.max_input_tokens,
        )

        for batch_idx, batch in enumerate(batches, start=1):
            logger.debug(
                "Processing fact extraction batch %d/%d (%d chunks)",
                batch_idx,
                len(batches),
                len(batch),
            )
            batch_facts = await self.llm_provider.extract_facts_batch(
                document_id=document_id,
                chunks=batch,
                metadata=metadata,
                on_dead_letter=on_dead_letter,
            )
            if not batch_facts:
                batch_cids = [c.chunk_id for c in batch]
                batch_headings = [c.path or " > ".join(c.heading_path) for c in batch]
                logger.info(
                    "[FACT_EXTRACTION_DIAGNOSTIC] Batch %d/%d returned 0 facts from %d chunks (ids=%s, headings=%s). "
                    "Inspected content evaluated as containing no actionable regulatory obligations.",
                    batch_idx,
                    len(batches),
                    len(batch),
                    batch_cids,
                    batch_headings,
                )
            all_facts.extend(batch_facts)

        logger.info(
            "Extracted %d total facts across %d batches for document %s",
            len(all_facts),
            len(batches),
            document_id,
        )
        return all_facts

