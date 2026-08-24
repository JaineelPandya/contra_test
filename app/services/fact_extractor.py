"""Fact extraction service coordinating batched LLM fact extraction with token-aware batching."""
from typing import Any, Callable, Dict, List, Optional
from app.core.config import get_settings
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact
from app.prompts.fact_extraction import BATCH_FACT_EXTRACTION_SYSTEM_PROMPT
from app.services.chunker import approximate_token_count


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

        # Dynamically create token-aware chunk batches
        batches = self.create_token_aware_batches(chunks, metadata)

        logger.info(
            "Extracting facts for document %s (%d total chunks packed into %d token-aware batches, budget=%d tokens)",
            document_id,
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
            all_facts.extend(batch_facts)

        logger.info(
            "Extracted %d total facts across %d batches for document %s",
            len(all_facts),
            len(batches),
            document_id,
        )
        return all_facts
