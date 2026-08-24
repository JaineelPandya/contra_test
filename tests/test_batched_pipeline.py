"""Integration tests for batched fact extraction, deterministic entity resolution, and provenance tracking."""
import pytest
from unittest.mock import MagicMock
from app.llm.mock_provider import MockProvider
from app.models.chunk import Chunk
from app.models.fact import Fact, FactPredicate
from app.services.fact_extractor import FactExtractor
from app.services.entity_resolver import EntityResolver


@pytest.mark.asyncio
async def test_fact_extractor_batches_chunks():
    provider = MockProvider()
    extractor = FactExtractor(provider)
    extractor.batch_size = 3

    # Generate 7 chunks
    chunks = [
        Chunk(
            chunk_id=f"chk-{i}",
            document_id="doc-test",
            content=f"Point {i}: Trading members must not share brokerage fees with unauthorized entities.",
            token_count=15,
            page_start=1,
            page_end=1,
            order=i,
        )
        for i in range(1, 8)
    ]

    facts = await extractor.extract_all_facts("doc-test", chunks)
    assert len(facts) >= 7
    # Verify chunks are traced back
    chunk_ids_in_facts = {f.chunk_id for f in facts}
    assert len(chunk_ids_in_facts) == 7


def test_deterministic_entity_resolution_no_llm():
    resolver = EntityResolver()

    facts = [
        Fact(
            fact_id="f1",
            document_id="doc-1",
            chunk_id="chk-1",
            subject="Trading Member",
            predicate=FactPredicate.PROHIBITS,
            object="Sharing fees",
            source_text="Trading members must not share fees.",
        ),
        Fact(
            fact_id="f2",
            document_id="doc-1",
            chunk_id="chk-2",
            subject="Securities and Exchange Board of India",
            predicate=FactPredicate.REQUIRES,
            object="Reporting",
            source_text="SEBI requires quarterly reporting.",
        ),
    ]

    chunks = [
        Chunk(
            chunk_id="chk-1",
            document_id="doc-1",
            content="Trading Member shall comply with National Stock Exchange rules.",
            token_count=10,
            page_start=1,
            page_end=1,
            order=0,
        )
    ]

    entities = resolver.resolve_from_facts(facts=facts, chunks=chunks)
    assert len(entities) >= 2

    canonical_names = {e.normalized_name for e in entities}
    assert "TRADING MEMBER" in canonical_names
    assert "SECURITIES AND EXCHANGE BOARD OF INDIA" in canonical_names
