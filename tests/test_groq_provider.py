"""Unit tests for Groq LLM provider, token-aware batching, TPM tracking, adaptive splitting, circuit breaker, and 429 long wait handling."""
import asyncio
from unittest.mock import patch, MagicMock
import httpx
import pytest
from app.llm.groq_provider import (
    GroqCircuitBreaker,
    GroqProvider,
    clean_json_response,
    parse_retry_after,
    sanitize_chunk_text,
    RollingTPMTracker,
)
from app.models.chunk import Chunk
from app.services.fact_extractor import FactExtractor


def test_sanitize_chunk_text() -> None:
    raw = "Point 1:\x00 Trading  \x01\x02members\tshall   maintain\n\n\n\nadequate margins."
    cleaned = sanitize_chunk_text(raw)
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert "\x02" not in cleaned
    assert "Trading members" in cleaned
    assert "adequate margins." in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_json_response() -> None:
    raw_markdown = '```json\n{"status": "ok"}\n```'
    assert clean_json_response(raw_markdown) == {"status": "ok"}


def test_clean_json_response_with_think_tags() -> None:
    raw_with_think = '<think>I need to process this and return JSON.</think>\n```json\n{"results": []}\n```'
    assert clean_json_response(raw_with_think) == {"results": []}


def test_clean_json_response_embedded_json() -> None:
    raw_embedded = 'Here is the output you requested:\n{"key": "value"}\nHope this helps!'
    assert clean_json_response(raw_embedded) == {"key": "value"}


def test_parse_retry_after_all_formats() -> None:
    # 1. Minutes + seconds e.g. "5m1.967999999s"
    tpd_msg = (
        "Rate limit reached for model `openai/gpt-oss-20b` on tokens per day (TPD): "
        "Limit 200000, Used 199433, Requested 1266. Please try again in 5m1.967999999s."
    )
    assert parse_retry_after(None, tpd_msg, default_backoff=5.0) == pytest.approx(301.968, 0.001)

    # 2. Large seconds format e.g. "1076s"
    tpm_long_msg = "Please try again in 1076s."
    assert parse_retry_after(None, tpm_long_msg, default_backoff=5.0) == 1076.0

    # 3. Fractional seconds format e.g. "49.0425s"
    tpm_msg = (
        "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): "
        "Limit: 8000, Used: 7976, Requested: 6563. Please try again in 49.0425s."
    )
    assert parse_retry_after(None, tpm_msg, default_backoff=5.0) == pytest.approx(49.0425, 0.001)

    # 4. Short seconds format e.g. "10s"
    short_msg = "Please try again in 10s."
    assert parse_retry_after(None, short_msg, default_backoff=5.0) == 10.0


def test_parse_retry_after_from_header() -> None:
    mock_resp = MagicMock()
    mock_resp.headers = {"retry-after": "15.5"}
    extracted = parse_retry_after(mock_resp, "error without time", default_backoff=5.0)
    assert extracted == 15.5


@pytest.mark.asyncio
async def test_groq_extract_facts_batch_success() -> None:
    provider = GroqProvider()

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"facts": ['
                        '{"subject": "Trading Member", "predicate": "REQUIRES", '
                        '"object": "Maintain separate bank account", '
                        '"regulatory_status": "MANDATORY", "confidence": 0.95, '
                        '"evidence": "Trading members must maintain separate bank accounts.", '
                        '"chunk_id": "chk-groq-1"}'
                        ']}'
                    )
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status.return_value = None

    chunk = Chunk(
        chunk_id="chk-groq-1",
        document_id="doc-groq-1",
        content="Trading members must maintain separate bank accounts.",
        token_count=8,
        page_start=1,
        page_end=1,
        order=0,
    )

    dead_letter_calls = []

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        facts = await provider.extract_facts_batch(
            "doc-groq-1",
            [chunk],
            on_dead_letter=lambda c, err, p: dead_letter_calls.append((c, err, p)),
        )
        assert len(facts) == 1
        assert facts[0].subject == "Trading Member"
        assert facts[0].predicate.value == "REQUIRES"
        assert facts[0].extraction_method == "groq"
        assert len(dead_letter_calls) == 0


@pytest.mark.asyncio
async def test_groq_429_short_wait_retries_and_succeeds() -> None:
    """When retry_after <= 60s, action is WAIT and retry."""
    provider = GroqProvider()

    error_429_text = (
        '{"error": {"message": "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): '
        'Limit: 8000, Used: 7900, Requested: 2000. Please try again in 10s."}}'
    )
    req_mock = MagicMock()
    resp_429 = httpx.Response(status_code=429, text=error_429_text, request=req_mock)
    error_429 = httpx.HTTPStatusError("429 Too Many Requests", request=req_mock, response=resp_429)

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"facts": [{"subject": "Member", "predicate": "REQUIRES", "object": "Maintain books", "confidence": 0.95, "evidence": "Trading members must maintain books.", "chunk_id": "chk-1"}]}'
                }
            }
        ]
    }
    resp_200.raise_for_status.return_value = None

    chunk = Chunk(chunk_id="chk-1", document_id="doc-1", content="Trading members must maintain books.", token_count=10)

    sleep_calls = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    with patch("httpx.AsyncClient.post", side_effect=[error_429, resp_200]) as mock_post, \
         patch("asyncio.sleep", side_effect=mock_sleep):

        facts = await provider.extract_facts_batch("doc-1", [chunk])

        assert mock_post.call_count == 2
        assert len(facts) == 1
        assert len(sleep_calls) == 1
        assert 12.0 <= sleep_calls[0] <= 15.0  # 10s + 2s buffer + jitter


@pytest.mark.asyncio
async def test_groq_429_long_wait_splits_batch_without_blocking() -> None:
    """When retry_after > 60s (e.g. 1100s), action is SPLIT without sleeping for 18 minutes."""
    provider = GroqProvider()

    error_429_long = (
        '{"error": {"message": "Rate limit reached. Please try again in 18m20s."}}'
    )
    req_mock = MagicMock()
    resp_429 = httpx.Response(status_code=429, text=error_429_long, request=req_mock)
    error_429 = httpx.HTTPStatusError("429 Too Many Requests", request=req_mock, response=resp_429)

    def make_200_resp(cid, subj):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": f'{{"facts": [{{"subject": "{subj}", "predicate": "REQUIRES", "object": "Task", "confidence": 0.95, "evidence": "Chunk {cid[-1]}", "chunk_id": "{cid}"}}]}}'
                    }
                }
            ]
        }
        resp.raise_for_status.return_value = None
        return resp

    chunk_a = Chunk(chunk_id="c-A", document_id="doc-1", content="Chunk A", token_count=10)
    chunk_b = Chunk(chunk_id="c-B", document_id="doc-1", content="Chunk B", token_count=10)
    chunk_c = Chunk(chunk_id="c-C", document_id="doc-1", content="Chunk C", token_count=10)

    sleep_calls = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        provider.tpm_tracker._usage_history.clear()

    # Batch [A, B, C] gets 429 (1100s) -> should NOT sleep 1100s -> splits into [A] and [B, C] which succeed
    with patch("httpx.AsyncClient.post", side_effect=[error_429, make_200_resp("c-A", "A"), make_200_resp("c-B", "B")]) as mock_post, \
         patch("asyncio.sleep", side_effect=mock_sleep):

        facts = await provider.extract_facts_batch("doc-1", [chunk_a, chunk_b, chunk_c])

        assert mock_post.call_count == 3
        assert len(facts) == 2
        # Verify worker did NOT sleep for 1100s
        for s in sleep_calls:
            assert s <= 65.0


@pytest.mark.asyncio
async def test_groq_429_long_wait_single_chunk_defers_to_fallback() -> None:
    """When retry_after > 60s on single chunk, action is DEFER / Fallback without blocking."""
    provider = GroqProvider()

    error_429_long = (
        '{"error": {"message": "Rate limit reached. Please try again in 5m1.967999999s."}}'
    )
    req_mock = MagicMock()
    resp_429 = httpx.Response(status_code=429, text=error_429_long, request=req_mock)
    error_429 = httpx.HTTPStatusError("429 Too Many Requests", request=req_mock, response=resp_429)

    chunk = Chunk(
        chunk_id="c-single",
        document_id="doc-1",
        content="Trading members shall not violate regulatory mandate under Section 8.",
        token_count=10,
    )

    dead_letter_calls = []
    sleep_calls = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        provider.tpm_tracker._usage_history.clear()

    with patch("httpx.AsyncClient.post", side_effect=error_429), \
         patch("asyncio.sleep", side_effect=mock_sleep):

        facts = await provider.extract_facts_batch(
            "doc-1",
            [chunk],
            on_dead_letter=lambda c, err, p: dead_letter_calls.append((c, err, p)),
        )

        # Accuracy-first: should record dead-letter with EXTRACTION_FAILED and NOT use rule-based fallback
        assert len(facts) == 0
        assert len(dead_letter_calls) == 1
        assert dead_letter_calls[0][0] == 429
        assert dead_letter_calls[0][2]["status"] == "EXTRACTION_FAILED"
        assert dead_letter_calls[0][2]["fallback_used"] is False
        for s in sleep_calls:
            assert s <= 305.0


@pytest.mark.asyncio
async def test_groq_circuit_breaker() -> None:
    """Circuit breaker trips after 3 consecutive 429s, cools down, and resets on success."""
    cb = GroqCircuitBreaker(failure_threshold=3, cooldown_seconds=10.0)

    assert cb.is_open()[0] is False

    cb.record_429()
    assert cb.is_open()[0] is False
    cb.record_429()
    assert cb.is_open()[0] is False
    cb.record_429()
    # 3rd failure trips the breaker
    is_open, remaining = cb.is_open()
    assert is_open is True
    assert 0 < remaining <= 10.0

    cb.record_success()
    assert cb.is_open()[0] is False


@pytest.mark.asyncio
async def test_adaptive_batch_splitting_on_400() -> None:
    provider = GroqProvider()

    req_mock = MagicMock()
    resp_400 = httpx.Response(
        status_code=400,
        text='{"error": {"code": "json_validate_failed", "message": "Failed to validate JSON."}}',
        request=req_mock,
    )
    error_400 = httpx.HTTPStatusError("400 Bad Request", request=req_mock, response=resp_400)

    def make_200_resp(cid, subj):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": f'{{"facts": [{{"subject": "{subj}", "predicate": "REQUIRES", "object": "Do duty", "confidence": 0.95, "evidence": "Content of chunk {cid[-1]}", "chunk_id": "{cid}"}}]}}'
                    }
                }
            ]
        }
        resp.raise_for_status.return_value = None
        return resp

    chunk_a = Chunk(chunk_id="chk-A", document_id="doc-split", content="Content of chunk A", token_count=10)
    chunk_b = Chunk(chunk_id="chk-B", document_id="doc-split", content="Content of chunk B", token_count=10)
    chunk_c = Chunk(chunk_id="chk-C", document_id="doc-split", content="Content of chunk C", token_count=10)

    async def mock_sleep(seconds: float) -> None:
        provider.tpm_tracker._usage_history.clear()

    with patch("httpx.AsyncClient.post", side_effect=[error_400, make_200_resp("chk-A", "Entity A"), make_200_resp("chk-B", "Entity B")]) as mock_post, \
         patch("asyncio.sleep", side_effect=mock_sleep):
        facts = await provider.extract_facts_batch("doc-split", [chunk_a, chunk_b, chunk_c])

        assert mock_post.call_count == 3
        assert len(facts) == 2


def test_token_aware_batching_splits_large_chunks() -> None:
    provider = GroqProvider()
    extractor = FactExtractor(provider)
    extractor.max_input_tokens = 1500

    chunk1 = Chunk(chunk_id="c1", document_id="d1", content="Paragraph 1", token_count=500)
    chunk2 = Chunk(chunk_id="c2", document_id="d1", content="Paragraph 2", token_count=600)
    chunk3 = Chunk(chunk_id="c3", document_id="d1", content="Paragraph 3", token_count=700)
    chunk4 = Chunk(chunk_id="c4", document_id="d1", content="Paragraph 4", token_count=400)

    batches = extractor.create_token_aware_batches([chunk1, chunk2, chunk3, chunk4])

    assert len(batches) >= 3


@pytest.mark.asyncio
async def test_rolling_tpm_tracker_safety_margin() -> None:
    """Tracker enforces safety margin (90% of limit)."""
    tracker = RollingTPMTracker(tpm_limit=5000, safety_margin=0.90)
    assert tracker.get_safety_limit() == 4500

    await tracker.acquire(3000)
    assert tracker.get_current_usage() == 3000

    sleep_calls = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        tracker._usage_history.clear()

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await tracker.acquire(2000)  # 3000 + 2000 = 5000 > 4500 safety limit

    assert len(sleep_calls) >= 1
    assert sleep_calls[0] > 0


def test_groq_model_configuration() -> None:
    provider = GroqProvider()
    assert provider.model == "openai/gpt-oss-20b"
    assert provider.tpm_limit == 8000
    assert provider.tpm_tracker.get_safety_limit() == 7200  # 90% of 8000
    assert provider.max_input_tokens == 4000
    assert provider.reserved_output_tokens == 1200
    assert provider.max_rate_limit_retries == 3
    assert provider.max_retry_wait_seconds == 300.0
    assert provider.rate_limit_safety_buffer == 2.0


