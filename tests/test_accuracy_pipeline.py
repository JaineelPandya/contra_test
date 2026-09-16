"""Comprehensive tests for the accuracy-first regulatory fact extraction pipeline:
- Batch size = 1 default
- Simplified atomic triple schema (subject, predicate, object)
- Python metadata enrichment (chunk_id, doc_id, page_number, provenance)
- Rejection of placeholders and empty fields
- Recovery cascade without rule-based fallback
- Groq error categorization (truncation vs schema error vs 429)
- Telemetry logging
- Second-pass verification stage (SUPPORTED, UNSUPPORTED, MODIFY, MISSING_FACT)
"""
import pytest
from unittest.mock import MagicMock, patch
import httpx
from app.llm.groq_provider import (
    GroqProvider,
    GroqJsonValidateError,
    GroqOutputTruncatedError,
    GroqRateLimitLongWaitError,
)
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact, FactPredicate


@pytest.fixture(autouse=True)
def mock_tpm_acquire():
    """Mock RollingTPMTracker.acquire to return immediately in unit tests."""
    with patch("app.llm.groq_provider.RollingTPMTracker.acquire", return_value=None):
        yield


@pytest.mark.asyncio
async def test_simplified_atomic_triple_enrichment() -> None:
    """LLM returns only {subject, predicate, object}; Python enriches all metadata."""
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-atomic-1",
        document_id="doc-123",
        parent_id="node-sec-1",
        page_start=5,
        page_end=5,
        content="Entities shall submit Form X within 15 days of the end of each month, provided that the threshold is met.",
        token_count=20,
    )
    metadata = DocumentMetadata(
        source="SEBI",
        department="SURVEILLANCE",
        document_date="2024-09-23",
        circular_number="CIR/2024/12",
        title="Master Circular",
    )

    llm_payload = {
        "facts": [
            {
                "subject": "Entities",
                "predicate": "shall submit",
                "object": "Form X",
            },
            {
                "subject": "Entities",
                "predicate": "shall submit Form X within",
                "object": "15 days of the end of each month",
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": str(llm_payload).replace("'", '"')}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 45}
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        facts = await provider.extract_facts_batch("doc-123", [chunk], metadata)

        assert len(facts) == 2
        # Verify Python-side metadata enrichment
        for f in facts:
            assert f.document_id == "doc-123"
            assert f.chunk_id == "chk-atomic-1"
            assert f.hierarchy_node_id == "node-sec-1"
            assert f.page_number == 5
            assert f.valid_from == "2024-09-23"
            assert f.extraction_method == "groq"
            assert f.predicate == FactPredicate.REQUIRES
            assert f.regulatory_status == "MANDATORY"
            assert "Entities shall submit Form X" in f.source_text
            # Automatic condition extraction from text
            assert any("provided that" in c.lower() for c in f.conditions)


def test_post_extraction_validation_rejects_placeholders_and_empty() -> None:
    """Post-extraction validation must reject empty fields and hallucinated placeholders."""
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-val-1",
        document_id="doc-val-1",
        content="Trading members shall maintain separate client bank accounts at all times.",
        token_count=12,
    )

    raw_facts = [
        # Valid
        {"subject": "Trading Members", "predicate": "shall maintain", "object": "separate client bank accounts"},
        # Empty subject
        {"subject": "", "predicate": "shall maintain", "object": "client bank accounts"},
        # Empty predicate
        {"subject": "Trading Members", "predicate": "", "object": "client bank accounts"},
        # Placeholder object
        {"subject": "Trading Members", "predicate": "shall maintain", "object": "string"},
        # Placeholder subject
        {"subject": "...", "predicate": "shall maintain", "object": "client bank accounts"},
        # Placeholder predicate
        {"subject": "Trading Members", "predicate": "N/A", "object": "client bank accounts"},
        # All identical tokens
        {"subject": "Trading", "predicate": "Trading", "object": "Trading"},
    ]

    valid = provider._validate_and_enrich_facts(
        raw_facts=raw_facts,
        document_id="doc-val-1",
        chunk=chunk,
        metadata=None,
        model_name="test-model",
    )

    assert len(valid) == 1
    assert valid[0].subject == "Trading Members"
    assert valid[0].object == "separate client bank accounts"


@pytest.mark.asyncio
async def test_groq_error_categorization_truncation() -> None:
    """Distinguish output truncation from schema validation failure."""
    provider = GroqProvider()
    error_text = (
        '{"error": {"code": "json_validate_failed", '
        '"message": "Failed to generate JSON. Please adjust your prompt.", '
        '"failed_generation": "max completion tokens reached before generating a valid document"}}'
    )
    req_mock = MagicMock()
    resp_400 = httpx.Response(status_code=400, text=error_text, request=req_mock)
    error_400 = httpx.HTTPStatusError("400 Bad Request", request=req_mock, response=resp_400)

    with patch("httpx.AsyncClient.post", side_effect=error_400):
        with pytest.raises(GroqOutputTruncatedError) as exc_info:
            await provider._generate_json(
                system_prompt="sys",
                user_prompt="user",
                max_output_tokens=600,
            )
        assert "max completion tokens reached" in exc_info.value.failed_generation


@pytest.mark.asyncio
async def test_cascade_escalates_to_alternate_model_and_succeeds() -> None:
    """When primary model fails JSON validation or truncation, cascade escalates to alternate model."""
    provider = GroqProvider()
    provider.alternate_model = "qwen/qwen3.8-27b"

    chunk = Chunk(
        chunk_id="chk-cascade-1",
        document_id="doc-casc-1",
        content="All market participants shall enrol on the ODR Portal within 30 days.",
        token_count=14,
    )

    req_mock = MagicMock()
    # 400 json_validate_failed error from primary model
    err_json_fail = httpx.Response(
        status_code=400,
        text='{"error": {"code": "json_validate_failed", "message": "Failed to validate JSON."}}',
        request=req_mock,
    )
    err_400 = httpx.HTTPStatusError("400 Bad Request", request=req_mock, response=err_json_fail)

    # Success response from alternate model (Stage 5)
    resp_success = MagicMock()
    resp_success.status_code = 200
    resp_success.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"facts": [{"subject": "Market Participants", "predicate": "shall enrol on", "object": "ODR Portal within 30 days"}]}'
            }
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30}
    }
    resp_success.raise_for_status.return_value = None

    # Fail stages 1, 2, 3, 4 (primary model), then succeed stage 5 (alternate model)
    with patch("httpx.AsyncClient.post", side_effect=[err_400, err_400, err_400, err_400, resp_success]) as mock_post:
        facts = await provider.extract_facts_batch("doc-casc-1", [chunk])

        assert len(facts) == 1
        assert facts[0].subject == "Market Participants"
        assert facts[0].model == "qwen/qwen3.8-27b"
        assert facts[0].extraction_method == "groq"
        assert mock_post.call_count == 5


@pytest.mark.asyncio
async def test_cascade_terminal_failure_records_dead_letter_without_fallback() -> None:
    """When all cascade attempts fail, mark EXTRACTION_FAILED and NEVER insert rule-based facts."""
    provider = GroqProvider()
    provider.alternate_model = "qwen/qwen3.8-27b"

    chunk = Chunk(
        chunk_id="chk-terminal-fail",
        document_id="doc-term-1",
        content="Complex unparseable regulatory text requiring compliance with Section 12.",
        token_count=10,
    )

    req_mock = MagicMock()
    err_resp = httpx.Response(
        status_code=400,
        text='{"error": {"code": "json_validate_failed", "message": "Persistent failure."}}',
        request=req_mock,
    )
    err_400 = httpx.HTTPStatusError("400 Bad Request", request=req_mock, response=err_resp)

    dead_letter_calls = []

    with patch("httpx.AsyncClient.post", side_effect=err_400):
        facts = await provider.extract_facts_batch(
            "doc-term-1",
            [chunk],
            on_dead_letter=lambda code, msg, payload: dead_letter_calls.append((code, msg, payload)),
        )

        # Accuracy-first requirement: NO rule-based fallback facts returned!
        assert len(facts) == 0
        assert len(dead_letter_calls) == 1
        code, msg, payload = dead_letter_calls[0]
        assert code == 400
        assert payload["status"] == "EXTRACTION_FAILED"
        assert payload["fallback_used"] is False
        assert payload["number_of_attempts"] >= 4
        assert payload["chunk_id"] == "chk-terminal-fail"


@pytest.mark.asyncio
async def test_second_pass_verification_flow() -> None:
    """Second-pass verifier audits candidate facts (retains SUPPORTED, applies MODIFY, drops UNSUPPORTED)."""
    provider = GroqProvider()
    provider.verify_second_pass = True

    chunk = Chunk(
        chunk_id="chk-verify-1",
        document_id="doc-v-1",
        content="Entities shall submit Form X within 15 days of month end, except entities below threshold.",
        token_count=18,
    )

    # Initial extraction response
    extract_resp = MagicMock()
    extract_resp.status_code = 200
    extract_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": (
                    '{"facts": ['
                    '{"subject": "Entities", "predicate": "shall submit", "object": "Form X"},'
                    '{"subject": "Entities", "predicate": "shall submit", "object": "Hallucinated Form Y"},'
                    '{"subject": "Entities", "predicate": "shall submit", "object": "within 15 days"}'
                    ']}'
                )
            }
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40}
    }
    extract_resp.raise_for_status.return_value = None

    # Verifier response
    verify_resp = MagicMock()
    verify_resp.status_code = 200
    verify_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": (
                    '{"verifications": ['
                    '{"fact_index": 0, "verdict": "SUPPORTED", "reason": "Directly stated"},'
                    '{"fact_index": 1, "verdict": "UNSUPPORTED", "reason": "Form Y not in text"},'
                    '{"fact_index": 2, "verdict": "MODIFY", "modified_fact": {"subject": "Entities", "predicate": "shall submit Form X within", "object": "15 days of month end"}, "reason": "Precision"}'
                    '],'
                    '"missing_facts": ['
                    '{"subject": "Entities below threshold", "predicate": "are exempt from", "object": "submission requirement", "reason": "Exception omitted"}'
                    ']}'
                )
            }
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 60}
    }
    verify_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", side_effect=[extract_resp, verify_resp]):
        verified_facts = await provider.extract_facts_batch("doc-v-1", [chunk])

        # Fact 0: SUPPORTED -> retained
        # Fact 1: UNSUPPORTED -> dropped
        # Fact 2: MODIFY -> updated
        # Missing fact recovered from verifier
        assert len(verified_facts) == 3
        subjects = [f.subject for f in verified_facts]
        assert "Entities" in subjects
        assert "Entities below threshold" in subjects
        # Form Y must be absent
        assert not any("Form Y" in f.object for f in verified_facts)


def test_directive_predicates_mapping() -> None:
    """Verifies that regulatory directives like 'directed to' map to REQUIRES and 'excludes' to EXCEPTION_TO."""
    provider = GroqProvider()

    # Directives -> REQUIRES / MANDATORY
    p1, s1 = provider._map_predicate_to_enum("are directed to put in place")
    assert p1 == FactPredicate.REQUIRES and s1 == "MANDATORY"

    p2, s2 = provider._map_predicate_to_enum("is directed to forward")
    assert p2 == FactPredicate.REQUIRES and s2 == "MANDATORY"

    p3, s3 = provider._map_predicate_to_enum("instructed to maintain")
    assert p3 == FactPredicate.REQUIRES and s3 == "MANDATORY"

    p4, s4 = provider._map_predicate_to_enum("prescribed under Rule 8")
    assert p4 == FactPredicate.REQUIRES and s4 == "MANDATORY"

    # Exclusions -> EXCEPTION_TO / ACTIVE
    p5, s5 = provider._map_predicate_to_enum("excludes")
    assert p5 == FactPredicate.EXCEPTION_TO and s5 == "ACTIVE"

    # Fabricated enums must continue to be rejected
    p_bad1, _ = provider._map_predicate_to_enum("MANDATES")
    assert p_bad1 is None

    p_bad2, _ = provider._map_predicate_to_enum("SUBSTITUTES_TOTALLY")
    assert p_bad2 is None


@pytest.mark.asyncio
async def test_circuit_breaker_and_429_cooperative_wait() -> None:
    """Cascade sleeps cooperatively on circuit breaker open or rate limit instead of fast-failing."""
    provider = GroqProvider()
    chunk = Chunk(chunk_id="chk-wait", document_id="doc-w", content="Market intermediaries shall adhere to surveillance norms.", token_count=10)

    sleep_calls = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        provider.tpm_tracker._usage_history.clear()

    # Mock stage 1: 429 with retry-after 10s -> cascade sleeps 12s -> stage 2 succeeds
    err_resp = httpx.Response(
        status_code=429,
        headers={"retry-after": "10"},
        text='{"error": {"message": "Rate limit reached. Please try again in 10s."}}',
        request=MagicMock(),
    )
    err_429 = httpx.HTTPStatusError("429 Too Many Requests", request=MagicMock(), response=err_resp)

    succ_resp = MagicMock()
    succ_resp.status_code = 200
    succ_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"facts": [{"subject": "Market intermediaries", "predicate": "REQUIRES", "object": "Adhere to norms", "confidence": 0.95, "evidence": "Market intermediaries shall adhere to surveillance norms."}]}'
            }
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30}
    }
    succ_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", side_effect=[err_429, succ_resp]), \
         patch("asyncio.sleep", side_effect=mock_sleep):
        facts = await provider._extract_chunk_with_cascade("doc-w", chunk, None)

    assert len(facts) == 1
    assert facts[0].subject == "Market intermediaries"
    assert any(s >= 10.0 for s in sleep_calls)
