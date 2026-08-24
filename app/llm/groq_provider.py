"""Groq LLM Provider implementation with token budgeting, circuit breaker, proactive TPM tracking, and adaptive batch degradation."""
import asyncio
import json
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
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
from app.services.chunker import approximate_token_count


class GroqRateLimitLongWaitError(Exception):
    """Raised when Groq returns HTTP 429 with a retry_after exceeding the allowed max wait."""

    def __init__(self, retry_after: float, message: str = "") -> None:
        self.retry_after = retry_after
        super().__init__(message or f"Groq 429 retry_after ({retry_after:.1f}s) exceeds maximum allowed wait.")


class GroqCircuitBreakerOpenError(Exception):
    """Raised when the Groq circuit breaker is active during a cooldown period."""

    def __init__(self, cooldown_remaining: float) -> None:
        self.cooldown_remaining = cooldown_remaining
        super().__init__(f"Groq circuit breaker is active. Cooldown remaining: {cooldown_remaining:.1f}s")


def sanitize_chunk_text(text: str) -> str:
    """Sanitize chunk text before sending to LLM:
    - Strip null bytes
    - Remove non-printable control characters (while preserving \\n, \\t, \\r)
    - Normalize excessive repeated whitespace while preserving newlines
    - Preserves section numbers, dates, circular references, percentages, etc.
    """
    if not text:
        return ""
    cleaned = text.replace("\x00", "")
    cleaned = "".join(
        ch for ch in cleaned
        if ch in ("\t", "\n", "\r") or (ord(ch) >= 32 and ord(ch) != 127)
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Strip markdown code fences, think tags, and clean JSON output."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if match:
            return json.loads(match.group(1))
        raise


def parse_retry_after(response: Optional[httpx.Response], error_text: str, default_backoff: float) -> float:
    """Extract retry duration in seconds from HTTP headers or response error message.

    Handles formats such as:
      - Header: 'Retry-After: 15.5'
      - Body: 'Please try again in 5m1.967999999s.' -> 301.968s
      - Body: 'Please try again in 49.0425s.' -> 49.0425s
      - Body: 'Please try again in 1191s.' -> 1191.0s
    """
    # 1. Check Retry-After header
    if response is not None and "retry-after" in response.headers:
        try:
            return float(response.headers["retry-after"])
        except (ValueError, TypeError):
            pass

    # 2. Check minutes + seconds format: "5m1.967s"
    m_min_sec = re.search(
        r"(?:try again in|retry after|in)\s+([0-9]+)m([0-9]+(?:\.[0-9]+)?)s",
        error_text,
        re.IGNORECASE,
    )
    if m_min_sec:
        try:
            minutes = float(m_min_sec.group(1))
            seconds = float(m_min_sec.group(2))
            return minutes * 60.0 + seconds
        except (ValueError, TypeError):
            pass

    # 3. Check seconds format: "49.0425s", "1100s", "10s"
    m_sec = re.search(
        r"(?:try again in|retry after|in)\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        error_text,
        re.IGNORECASE,
    )
    if m_sec:
        try:
            return float(m_sec.group(1))
        except (ValueError, TypeError):
            pass

    return default_backoff


class RollingTPMTracker:
    """Tracks token usage over a rolling 60-second window with safety margin enforcement."""

    def __init__(self, tpm_limit: int = 8000, safety_margin: float = 0.90) -> None:
        self.tpm_limit = tpm_limit
        self.safety_margin = safety_margin
        self.safety_limit = int(tpm_limit * safety_margin)
        self._usage_history: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()

    def get_current_usage(self) -> int:
        """Return token count consumed within the current 60s rolling window."""
        now = time.monotonic()
        return sum(tokens for ts, tokens in self._usage_history if now - ts < 60.0)

    def get_safety_limit(self) -> int:
        """Return the effective TPM safety limit."""
        return self.safety_limit

    async def acquire(self, estimated_tokens: int) -> None:
        """Check rolling token budget and pause asynchronously if safety limit would be exceeded."""
        while True:
            wait_seconds = 0.0
            async with self._lock:
                now = time.monotonic()
                self._usage_history = [
                    (ts, tokens) for ts, tokens in self._usage_history if now - ts < 60.0
                ]
                current_usage = sum(tokens for _, tokens in self._usage_history)

                if current_usage + estimated_tokens <= self.safety_limit:
                    self._usage_history.append((now, estimated_tokens))
                    return

                oldest_ts = self._usage_history[0][0]
                wait_seconds = max(0.5, (oldest_ts + 60.0) - now + 0.5)
                logger.info(
                    "Local TPM throttle: %d tokens used in last 60s + %d requested > safety limit %d (raw limit %d). "
                    "Waiting %.2fs for rolling window to clear...",
                    current_usage,
                    estimated_tokens,
                    self.safety_limit,
                    self.tpm_limit,
                    wait_seconds,
                )

            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)


class GroqCircuitBreaker:
    """Circuit breaker preventing request hammering during persistent 429 rate limit periods."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_429_count: int = 0
        self.cooldown_until: float = 0.0

    def is_open(self) -> Tuple[bool, float]:
        """Check if circuit breaker is currently open. Returns (is_open, remaining_cooldown)."""
        now = time.monotonic()
        if now < self.cooldown_until:
            return True, self.cooldown_until - now
        return False, 0.0

    def record_success(self) -> None:
        """Reset consecutive failure count and cooldown upon successful request."""
        self.consecutive_429_count = 0
        self.cooldown_until = 0.0

    def record_429(self) -> None:
        """Increment failure counter and trip breaker if threshold exceeded."""
        self.consecutive_429_count += 1
        if self.consecutive_429_count >= self.failure_threshold:
            self.cooldown_until = time.monotonic() + self.cooldown_seconds
            logger.warning(
                "Groq circuit breaker TRIPPED after %d consecutive 429 errors. "
                "Cooling down for %.1fs before sending new requests.",
                self.consecutive_429_count,
                self.cooldown_seconds,
            )


class GroqProvider(BaseLLMProvider):
    """LLM Provider interfacing with Groq API with robust token budgeting, circuit breaker, and adaptive splitting."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.GROQ_BASE_URL.rstrip("/")
        self.api_key = self.settings.GROQ_API_KEY
        self.model = self.settings.GROQ_MODEL
        self.temperature = self.settings.GROQ_TEMPERATURE
        self.timeout = 45.0

        self.tpm_limit = self.settings.GROQ_TPM_LIMIT
        self.tpm_safety_margin = self.settings.GROQ_TPM_SAFETY_MARGIN
        self.max_input_tokens = self.settings.GROQ_MAX_REQUEST_INPUT_TOKENS
        self.reserved_output_tokens = self.settings.GROQ_RESERVED_OUTPUT_TOKENS
        self.max_rate_limit_retries = self.settings.GROQ_MAX_RATE_LIMIT_RETRIES
        self.max_retry_wait_seconds = self.settings.GROQ_MAX_RETRY_WAIT_SECONDS
        self.rate_limit_safety_buffer = self.settings.GROQ_RATE_LIMIT_SAFETY_BUFFER_SECONDS

        self.tpm_tracker = RollingTPMTracker(
            tpm_limit=self.tpm_limit,
            safety_margin=self.tpm_safety_margin,
        )
        self.circuit_breaker = GroqCircuitBreaker(
            failure_threshold=self.settings.GROQ_CIRCUIT_BREAKER_CONSECUTIVE_FAILURES,
            cooldown_seconds=self.settings.GROQ_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set the GROQ_API_KEY environment variable."
            )
        logger.info(
            "Groq configured model: %s | TPM limit: %d (safety: %d) | reserved output: %d | max wait: %.1fs",
            self.model,
            self.tpm_limit,
            self.tpm_tracker.get_safety_limit(),
            self.reserved_output_tokens,
            self.max_retry_wait_seconds,
        )

    async def _generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        context_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to Groq API using json_object mode, token budgeting, and bounded 429 retries."""
        # 1. Proactive circuit breaker check
        is_open, remaining_cooldown = self.circuit_breaker.is_open()
        if is_open:
            logger.warning(
                "Groq request blocked by circuit breaker (cooldown remaining: %.1fs) | %s",
                remaining_cooldown,
                context_info or "",
            )
            raise GroqCircuitBreakerOpenError(remaining_cooldown)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        output_budget = max_output_tokens or self.reserved_output_tokens
        sys_tokens = approximate_token_count(system_prompt)
        user_tokens = approximate_token_count(user_prompt)
        total_input_tokens = sys_tokens + user_tokens
        total_estimated_request_cost = total_input_tokens + output_budget

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": output_budget,
        }

        current_tpm_used = self.tpm_tracker.get_current_usage()
        tpm_safety_limit = self.tpm_tracker.get_safety_limit()

        # Detailed logging for every Groq request token budget
        logger.info(
            "Groq request token budget | model=%s | est_input_tokens=%d | max_output_tokens=%d | "
            "total_req_tokens=%d | TPM_used=%d | TPM_safety_limit=%d | %s",
            self.model,
            total_input_tokens,
            output_budget,
            total_estimated_request_cost,
            current_tpm_used,
            tpm_safety_limit,
            context_info or "",
        )

        # 2. Local TPM Proactive Throttling
        await self.tpm_tracker.acquire(total_estimated_request_cost)

        # 3. Retry loop for HTTP 429
        attempt = 0
        total_allowed_attempts = self.max_rate_limit_retries + 1

        while attempt < total_allowed_attempts:
            attempt += 1
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    self.circuit_breaker.record_success()
                    return clean_json_response(content)

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    error_body = e.response.text

                    if status_code == 429:
                        self.circuit_breaker.record_429()
                        backoff_fallback = min(self.max_retry_wait_seconds, 5.0 * (2 ** (attempt - 1)))
                        retry_after = parse_retry_after(e.response, error_body, backoff_fallback)

                        # Decision: WAIT or REJECT LONG WAIT
                        if retry_after <= self.max_retry_wait_seconds and attempt < total_allowed_attempts:
                            jitter = random.uniform(0.5, 2.0)
                            sleep_duration = min(
                                self.max_retry_wait_seconds,
                                retry_after + self.rate_limit_safety_buffer + jitter,
                            )
                            logger.warning(
                                "Groq 429 Decision | action=WAIT | server_retry_after=%.2fs <= max_wait=%.2fs | "
                                "sleep_duration=%.2fs | attempt=%d/%d | TPM_used=%d | TPM_safety_limit=%d | req_tokens=%d | %s",
                                retry_after,
                                self.max_retry_wait_seconds,
                                sleep_duration,
                                attempt,
                                total_allowed_attempts,
                                self.tpm_tracker.get_current_usage(),
                                tpm_safety_limit,
                                total_estimated_request_cost,
                                context_info or "",
                            )
                            await asyncio.sleep(sleep_duration)
                            continue
                        else:
                            # Long wait: DO NOT block worker for minutes/hours!
                            logger.warning(
                                "Groq 429 Decision | action=SPLIT/DEFER | server_retry_after=%.2fs > max_wait=%.2fs | "
                                "worker_not_blocked | attempt=%d/%d | TPM_used=%d | TPM_safety_limit=%d | req_tokens=%d | %s",
                                retry_after,
                                self.max_retry_wait_seconds,
                                attempt,
                                total_allowed_attempts,
                                self.tpm_tracker.get_current_usage(),
                                tpm_safety_limit,
                                total_estimated_request_cost,
                                context_info or "",
                            )
                            raise GroqRateLimitLongWaitError(retry_after=retry_after)

                    # Diagnostic logging for HTTP 400
                    failed_gen = ""
                    error_code = ""
                    error_msg = error_body
                    try:
                        err_json = json.loads(error_body)
                        err_data = err_json.get("error", {})
                        failed_gen = err_data.get("failed_generation", "")
                        error_code = err_data.get("code", "")
                        error_msg = err_data.get("message", error_body)
                    except Exception:
                        pass

                    logger.error(
                        "Groq HTTP failure | status=%s | code=%s | message=%s | failed_generation=%s | attempt=%d/%d | %s",
                        status_code,
                        error_code,
                        error_msg,
                        failed_gen[:200] if failed_gen else "<empty>",
                        attempt,
                        total_allowed_attempts,
                        context_info or "",
                    )
                    raise

                except (GroqRateLimitLongWaitError, GroqCircuitBreakerOpenError):
                    raise
                except Exception as e:
                    logger.error("Groq API request exception (attempt %d/%d): %s", attempt, total_allowed_attempts, e)
                    raise

        raise RuntimeError(f"Groq request failed after {total_allowed_attempts} attempts.")

    def _parse_raw_facts(
        self,
        data: Dict[str, Any],
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata],
    ) -> List[Fact]:
        """Parse and validate facts from JSON response safely."""
        raw_facts = data.get("facts", [])
        if not raw_facts and "results" in data:
            for r in data.get("results", []):
                raw_facts.extend(r.get("facts", []))

        facts: List[Fact] = []
        chunk_map = {c.chunk_id: c for c in chunks}

        for rf in raw_facts:
            try:
                cid = rf.get("chunk_id")
                matched_chunk = chunk_map.get(cid) if cid else None
                if not matched_chunk and chunks:
                    matched_chunk = chunks[0]
                    cid = chunks[0].chunk_id

                pred_str = str(rf.get("predicate", "REQUIRES")).upper()
                try:
                    predicate = FactPredicate(pred_str)
                except ValueError:
                    predicate = FactPredicate.REQUIRES

                reg_status = "ACTIVE"
                if predicate == FactPredicate.PROHIBITS:
                    reg_status = "PROHIBITED"
                elif predicate == FactPredicate.PERMITS:
                    reg_status = "PERMITTED"
                elif predicate == FactPredicate.DELETES:
                    reg_status = "DELETED"
                elif predicate == FactPredicate.REQUIRES:
                    reg_status = "MANDATORY"

                source_text = (
                    rf.get("evidence")
                    or rf.get("source_text")
                    or (matched_chunk.content[:200] if matched_chunk else "")
                )

                fact = Fact(
                    document_id=document_id,
                    chunk_id=cid if cid else chunks[0].chunk_id,
                    hierarchy_node_id=matched_chunk.parent_id if matched_chunk else None,
                    subject=rf.get("subject", "Regulated Entity"),
                    predicate=predicate,
                    object=rf.get("object", ""),
                    conditions=rf.get("conditions", []),
                    exceptions=rf.get("exceptions", []),
                    regulatory_status=rf.get("regulatory_status", reg_status),
                    effective_date=rf.get("effective_date"),
                    confidence=float(rf.get("confidence", 0.95)),
                    source_text=source_text,
                    extraction_method="groq",
                    model=self.model,
                    status=FactStatus.ACTIVE,
                    valid_from=metadata.document_date if metadata else None,
                )
                facts.append(fact)
            except Exception as fact_err:
                logger.warning("Skipping malformed fact entry in Groq response: %s (error: %s)", rf, fact_err)

        return facts

    async def extract_facts_batch(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
        """Extract atomic regulatory facts with content sanitization and adaptive recursive batch degradation."""
        if not chunks:
            return []

        # 1. Content Sanitization
        sanitized_chunks = [
            Chunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                parent_id=c.parent_id,
                heading_path=c.heading_path,
                content=sanitize_chunk_text(c.content),
                token_count=c.token_count,
                page_start=c.page_start,
                page_end=c.page_end,
                order=c.order,
            )
            for c in chunks
        ]

        chunks_payload = [
            {
                "chunk_id": c.chunk_id,
                "heading_path": " > ".join(c.heading_path) if c.heading_path else "Section",
                "text": c.content,
            }
            for c in sanitized_chunks
        ]

        meta_json = metadata.model_dump_json() if metadata else "{}"
        user_prompt = BATCH_FACT_EXTRACTION_USER_PROMPT.format(
            metadata_json=meta_json,
            chunks_json=json.dumps(chunks_payload, indent=2),
        )

        chunk_ids = [c.chunk_id for c in sanitized_chunks]
        data = None
        rate_limit_long_wait = False
        circuit_breaker_active = False

        # 2. Try primary extraction
        try:
            data = await self._generate_json(
                system_prompt=BATCH_FACT_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=self.reserved_output_tokens,
                context_info=f"batch of {len(sanitized_chunks)} chunks ({','.join(chunk_ids)})",
            )
        except GroqRateLimitLongWaitError as rle:
            rate_limit_long_wait = True
            logger.warning("Fact extraction received 429 long wait (%.1fs). Triggering degradation strategy...", rle.retry_after)
        except GroqCircuitBreakerOpenError as cbe:
            circuit_breaker_active = True
            logger.warning("Fact extraction halted by circuit breaker (cooldown: %.1fs). Triggering degradation strategy...", cbe.cooldown_remaining)
        except httpx.HTTPStatusError as http_err:
            if http_err.response.status_code == 400:
                logger.warning("Fact extraction received HTTP 400 (json_validate_failed). Triggering adaptive batch split...")
            else:
                logger.error("Groq HTTP error %d: %s", http_err.response.status_code, http_err.response.text[:200])
        except Exception as gen_err:
            logger.error("Fact extraction exception: %s", gen_err)

        # 3. If primary attempt succeeded, parse and return facts
        if data and ("facts" in data or "results" in data):
            facts = self._parse_raw_facts(data, document_id, sanitized_chunks, metadata)
            logger.info("Successfully extracted %d facts via Groq for batch of %d chunks", len(facts), len(sanitized_chunks))
            return facts

        # 4. Adaptive Batch Splitting / Degradation Strategy
        if len(sanitized_chunks) > 1:
            # Action: SPLIT batch into smaller subgroups
            mid = len(sanitized_chunks) // 2
            first_half = sanitized_chunks[:mid]
            second_half = sanitized_chunks[mid:]
            logger.warning(
                "Adaptive Batch Split | action=SPLIT | reason=%s | batch_size=%d -> (%d, %d) | chunk_ids=%s",
                "429_long_wait" if rate_limit_long_wait else ("circuit_breaker" if circuit_breaker_active else "json_validate_failed"),
                len(sanitized_chunks),
                len(first_half),
                len(second_half),
                chunk_ids,
            )
            facts_first = await self.extract_facts_batch(document_id, first_half, metadata, on_dead_letter)
            facts_second = await self.extract_facts_batch(document_id, second_half, metadata, on_dead_letter)
            return facts_first + facts_second
        else:
            # Single isolated chunk
            single_chunk = sanitized_chunks[0]
            cid = single_chunk.chunk_id
            content_str = single_chunk.content

            # If failure was a 429 long wait or circuit breaker on a single chunk:
            # Action: DEFER / Fallback without blocking the worker for minutes
            if rate_limit_long_wait or circuit_breaker_active:
                logger.warning(
                    "Adaptive Degradation | action=DEFER | reason=429_long_wait_single_chunk | "
                    "chunk_id=%s | worker_not_blocked | using rule-based fallback",
                    cid,
                )
                if on_dead_letter:
                    on_dead_letter(
                        429,
                        f"Groq 429 rate limit long wait for isolated chunk {cid}; deferred to rule-based fallback without blocking worker.",
                        {
                            "provider": "groq",
                            "model": self.model,
                            "status_code": 429,
                            "chunk_id": cid,
                            "char_count": len(content_str),
                            "preview": content_str[:200],
                            "fallback_used": True,
                        },
                    )
                return self._fallback_fact_extraction(document_id, [single_chunk], metadata)

            # If failure was HTTP 400 json_validate_failed on a single chunk:
            # Try minimal single-chunk recovery prompt once
            try:
                recovery_sys = (
                    "You are a regulatory fact extraction assistant. "
                    "Extract atomic facts from this chunk and return ONLY a JSON object matching: "
                    "{\"facts\": [{\"subject\": \"...\", \"predicate\": \"REQUIRES\", \"object\": \"...\", \"evidence\": \"...\", \"confidence\": 0.95, \"chunk_id\": \"...\"}]}."
                )
                recovery_user = f"Extract regulatory facts from chunk ID '{cid}':\n{content_str}"
                recovery_data = await self._generate_json(
                    system_prompt=recovery_sys,
                    user_prompt=recovery_user,
                    max_output_tokens=self.reserved_output_tokens,
                    context_info=f"single chunk recovery ({cid})",
                )
                if recovery_data:
                    recovery_facts = self._parse_raw_facts(recovery_data, document_id, [single_chunk], metadata)
                    if recovery_facts:
                        logger.info("Single chunk recovery succeeded for %s (%d facts)", cid, len(recovery_facts))
                        return recovery_facts
            except Exception as single_err:
                logger.error("Single chunk minimal recovery also failed for %s: %s", cid, single_err)

            # If minimal recovery failed, fallback only for this single chunk
            if on_dead_letter:
                on_dead_letter(
                    400,
                    f"Groq fact extraction failed for isolated chunk {cid} after minimal recovery attempt.",
                    {
                        "provider": "groq",
                        "model": self.model,
                        "status_code": 400,
                        "chunk_id": cid,
                        "char_count": len(content_str),
                        "preview": content_str[:200],
                        "fallback_used": True,
                    },
                )

            logger.warning("Using rule-based fallback ONLY for isolated failing chunk %s", cid)
            return self._fallback_fact_extraction(document_id, [single_chunk], metadata)

    async def extract_metadata(self, header_text: str) -> Optional[DocumentMetadata]:
        """Extract structured document metadata using Groq."""
        try:
            user_prompt = METADATA_EXTRACTION_USER_PROMPT.format(header_text=header_text)
            data = await self._generate_json(
                METADATA_EXTRACTION_SYSTEM_PROMPT,
                user_prompt,
                max_output_tokens=600,
                context_info="metadata extraction",
            )
            return DocumentMetadata(**data)
        except Exception as e:
            logger.warning("Groq metadata extraction failed: %s", e)
            return None

    async def classify_relationships_batch(
        self,
        candidate_pairs: List[Dict[str, Any]],
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of candidate relationship pairs using Groq."""
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

            data = await self._generate_json(
                system_prompt=BATCH_RELATIONSHIP_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=self.reserved_output_tokens,
                context_info=f"batch of {len(candidate_pairs)} candidate pairs",
            )
            results = data.get("results", [])
            for r in results:
                r["extraction_method"] = "groq"
                r["model"] = self.model
            return results

        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", 500)
            err_details = getattr(getattr(e, "response", None), "text", str(e))
            logger.warning(
                "Groq batched relationship classification failed (status=%s, error=%s)",
                status_code,
                err_details,
            )
            if on_dead_letter:
                on_dead_letter(
                    status_code,
                    f"Groq API Error ({status_code}): {err_details}",
                    {
                        "provider": "groq",
                        "model": self.model,
                        "status_code": status_code,
                        "raw_error": err_details,
                        "pairs_count": len(candidate_pairs),
                        "candidate_pairs": candidate_pairs,
                    },
                )
            return []

    async def detect_references(
        self,
        source_document_id: str,
        text: str,
    ) -> List[DocumentReference]:
        """Detect citations and cross-document references from text using Groq."""
        try:
            user_prompt = REFERENCE_DETECTION_USER_PROMPT.format(text=text)
            data = await self._generate_json(
                system_prompt=REFERENCE_DETECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=600,
                context_info="reference detection",
            )
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
            logger.warning("Groq reference detection failed: %s", e)
            return []

    def _fallback_fact_extraction(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
    ) -> List[Fact]:
        """Deterministic rule-based fallback if Groq API call fails."""
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
                    fallback_reason="groq_fallback",
                    status=FactStatus.ACTIVE,
                    valid_from=metadata.document_date if metadata else None,
                )
                facts.append(fact)

        return facts
