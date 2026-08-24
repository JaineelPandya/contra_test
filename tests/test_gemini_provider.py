"""Unit tests for Google Gemini LLM provider with batching and request manager."""
from unittest.mock import patch, MagicMock
import pytest
from app.llm.gemini_provider import GeminiProvider, clean_json_response
from app.models.chunk import Chunk


def test_clean_json_response() -> None:
    raw_markdown = '```json\n{"status": "ok"}\n```'
    assert clean_json_response(raw_markdown) == {"status": "ok"}


@pytest.mark.asyncio
async def test_gemini_classify_relationships_batch() -> None:
    provider = GeminiProvider()

    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"results": [{"source_id": "f1", "target_id": "f2", "relationship": "MODIFIES", "confidence": 0.95, "reason": "Clause modified"}]}'
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_response
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        results = await provider.classify_relationships_batch(
            candidate_pairs=[{"source_id": "f1", "target_id": "f2", "source_text": "Amended", "target_text": "Rule"}]
        )

        assert len(results) == 1
        assert results[0]["relationship"] == "MODIFIES"
        assert results[0]["source_id"] == "f1"
        assert results[0]["extraction_method"] == "gemini"


@pytest.mark.asyncio
async def test_gemini_extract_facts_batch() -> None:
    provider = GeminiProvider()

    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"results": [{"chunk_id": "chk-1", "facts": [{"subject": "Trading Member", "predicate": "PROHIBITS", "object": "Sharing fees", "regulatory_status": "PROHIBITED", "confidence": 0.95, "source_text": "Sharing fees is prohibited."}]}]}'
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_response
    mock_resp.raise_for_status.return_value = None

    chunk = Chunk(
        chunk_id="chk-1",
        document_id="doc-1",
        content="Sharing fees is prohibited.",
        token_count=5,
        page_start=1,
        page_end=1,
        order=0,
    )

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        facts = await provider.extract_facts_batch("doc-1", [chunk])
        assert len(facts) == 1
        assert facts[0].subject == "Trading Member"
        assert facts[0].predicate.value == "PROHIBITS"
        assert facts[0].extraction_method == "gemini"
