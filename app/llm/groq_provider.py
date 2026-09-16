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
from app.models.fact import Fact, FactPredicate, FactStatus, normalize_text
from app.models.reference import DocumentReference
from app.prompts.metadata import METADATA_EXTRACTION_SYSTEM_PROMPT, METADATA_EXTRACTION_USER_PROMPT
from app.prompts.fact_extraction import (
    ACCURACY_FIRST_FACT_EXTRACTION_SYSTEM_PROMPT,
    ACCURACY_FIRST_FACT_EXTRACTION_USER_PROMPT,
    BATCH_FACT_EXTRACTION_SYSTEM_PROMPT,
    BATCH_FACT_EXTRACTION_USER_PROMPT,
    SIMPLIFIED_FALLBACK_FACT_EXTRACTION_SYSTEM_PROMPT,
    SIMPLIFIED_FALLBACK_FACT_EXTRACTION_USER_PROMPT,
    FACT_VERIFICATION_SYSTEM_PROMPT,
    FACT_VERIFICATION_USER_PROMPT,
)
from app.prompts.change_detection import BATCH_RELATIONSHIP_SYSTEM_PROMPT, BATCH_RELATIONSHIP_USER_PROMPT
from app.prompts.reference_detection import REFERENCE_DETECTION_SYSTEM_PROMPT, REFERENCE_DETECTION_USER_PROMPT
from app.services.qualifier_extractor import extract_qualifier_spans


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


class GroqJsonValidateError(Exception):
    """Raised when Groq returns HTTP 400 json_validate_failed."""

    def __init__(self, message: str = "", failed_generation: str = "") -> None:
        self.failed_generation = failed_generation
        super().__init__(message or f"Groq JSON validation failed: {failed_generation}")


class GroqOutputTruncatedError(Exception):
    """Raised when Groq output is truncated before completing valid JSON."""

    def __init__(self, message: str = "", failed_generation: str = "") -> None:
        self.failed_generation = failed_generation
        super().__init__(message or f"Groq output token limit reached: {failed_generation}")


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
                # Safeguard against mock environments where time.monotonic() is not advanced
                if time.monotonic() <= now:
                    async with self._lock:
                        if self._usage_history:
                            self._usage_history.pop(0)


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

        self.max_output_tokens_retry = getattr(self.settings, "GROQ_MAX_OUTPUT_TOKENS_RETRY", 2000)
        self.alternate_model = getattr(self.settings, "GROQ_ALTERNATE_MODEL", "qwen/qwen3.8-27b")
        self.verify_second_pass = getattr(self.settings, "FACT_EXTRACTION_VERIFY_SECOND_PASS", False)

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set the GROQ_API_KEY environment variable."
            )
        logger.info(
            "Groq configured model: %s (alt: %s) | TPM limit: %d (safety: %d) | reserved output: %d (retry: %d) | max wait: %.1fs",
            self.model,
            self.alternate_model,
            self.tpm_limit,
            self.tpm_tracker.get_safety_limit(),
            self.reserved_output_tokens,
            self.max_output_tokens_retry,
            self.max_retry_wait_seconds,
        )

    async def _generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        context_info: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        return_usage: bool = False,
    ) -> Any:
        """Send chat completion request to Groq API using json_object mode, token budgeting, and bounded 429 retries."""
        # 1. Proactive circuit breaker check: cooperative asynchronous wait instead of fast-failing
        is_open, remaining_cooldown = self.circuit_breaker.is_open()
        if is_open:
            sleep_wait = remaining_cooldown + 1.0
            logger.warning(
                "Groq circuit breaker open (cooldown remaining: %.1fs). "
                "Cooperatively pausing worker for %.1fs before request to clear cooldown | %s",
                remaining_cooldown,
                sleep_wait,
                context_info or "",
            )
            await asyncio.sleep(sleep_wait)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req_model = model or self.model
        req_temp = self.temperature if temperature is None else temperature
        output_budget = max_output_tokens or self.reserved_output_tokens
        sys_tokens = approximate_token_count(system_prompt)
        user_tokens = approximate_token_count(user_prompt)
        total_input_tokens = sys_tokens + user_tokens
        total_estimated_request_cost = total_input_tokens + output_budget

        payload: Dict[str, Any] = {
            "model": req_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": req_temp,
            "max_tokens": output_budget,
        }

        current_tpm_used = self.tpm_tracker.get_current_usage()
        tpm_safety_limit = self.tpm_tracker.get_safety_limit()

        # Detailed logging for every Groq request token budget
        logger.info(
            "Groq request token budget | model=%s | est_input_tokens=%d | max_output_tokens=%d | "
            "total_req_tokens=%d | TPM_used=%d | TPM_safety_limit=%d | %s",
            req_model,
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
                    usage = data.get("usage", {})
                    self.circuit_breaker.record_success()
                    parsed = clean_json_response(content)
                    if return_usage:
                        return parsed, usage
                    return parsed

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

                    if status_code == 400:
                        failed_lower = (failed_gen or "").lower()
                        msg_lower = (error_msg or "").lower()
                        if "max completion tokens reached" in failed_lower or "max completion tokens reached" in msg_lower or "length" in failed_lower:
                            raise GroqOutputTruncatedError(error_msg, failed_generation=failed_gen)
                        elif error_code == "json_validate_failed" or "json" in msg_lower or "validate" in msg_lower:
                            raise GroqJsonValidateError(error_msg, failed_generation=failed_gen)

                    raise

                except (GroqRateLimitLongWaitError, GroqCircuitBreakerOpenError, GroqOutputTruncatedError, GroqJsonValidateError):
                    raise
                except Exception as e:
                    logger.error("Groq API request exception (attempt %d/%d): %s", attempt, total_allowed_attempts, e)
                    raise

        raise RuntimeError(f"Groq request failed after {total_allowed_attempts} attempts.")

    def _find_grounded_sentence(self, chunk_text: str, subject: str, obj: str) -> str:
        """Find the best verbatim sentence in chunk_text for provenance grounding."""
        if not chunk_text or not chunk_text.strip():
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return chunk_text.strip()[:350]

        stop_words = {"the", "a", "an", "of", "and", "or", "in", "to", "for", "with", "on", "at", "by", "from", "as", "is", "are", "be"}

        def extract_kw(t: str) -> set:
            return {w.lower() for w in re.findall(r"\b[A-Za-z0-9]+\b", t) if w.lower() not in stop_words}

        query_kw = extract_kw(subject) | extract_kw(obj)
        best_sentence = sentences[0]
        best_overlap = -1

        for s in sentences:
            overlap = len(query_kw & extract_kw(s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s

        if normalize_text(best_sentence) in normalize_text(chunk_text):
            return best_sentence
        return chunk_text.strip()[:350]

    def _map_predicate_to_enum(self, pred_str: str) -> Tuple[Optional[FactPredicate], Optional[str]]:
        """Map raw predicate string to canonical FactPredicate and regulatory requirement state."""
        pred_stripped = pred_str.strip()
        pred_upper = pred_stripped.upper()

        if pred_upper in FactPredicate.__members__:
            pred = FactPredicate(pred_upper)
            if pred == FactPredicate.PROHIBITS:
                return pred, "PROHIBITED"
            elif pred == FactPredicate.PERMITS:
                return pred, "PERMITTED"
            elif pred in (FactPredicate.DELETES, FactPredicate.REPEALS):
                return pred, "DELETED"
            elif pred in (FactPredicate.MODIFIES, FactPredicate.AMENDS, FactPredicate.REPLACES, FactPredicate.SUPERSEDES):
                return pred, "MODIFIED"
            elif pred in (
                FactPredicate.REQUIRES,
                FactPredicate.CREDITED_TO,
                FactPredicate.PAYMENT_DEADLINE,
                FactPredicate.SUBMITS_TO,
                FactPredicate.REPORTS_TO,
                FactPredicate.MAINTAINS,
                FactPredicate.IMPLEMENTS,
            ):
                return pred, "MANDATORY"
            return pred, "ACTIVE"

        # If it looks like a fabricated uppercase enum (all uppercase with underscores or letters, but not valid), reject
        if pred_stripped == pred_upper and ("_" in pred_stripped or len(pred_stripped) > 4):
            return None, None

        clean_pred = re.sub(r"[_\-]+", " ", pred_str).strip()

        if re.search(r"\b(shall\s+not|must\s+not|shall\s+neither|neither\b.*\bnor|prohibit\w*|cannot|barred|forbidden|disallow\w*|restrict\w*|refrain|non-compliance)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.PROHIBITS, "PROHIBITED"
        elif re.search(r"\b(may|permit\w*|allowed|entitled\s+to|discretion|authoriz\w*|enable\w*|free\s+to|at\s+liberty)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.PERMITS, "PERMITTED"
        elif re.search(r"\b(exempt\w*|exception\w*|shall\s+not\s+apply|not\s+applicable|carve-out|exclude\w*|excluding)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.EXCEPTION_TO, "ACTIVE"
        elif re.search(r"\b(substitut\w*|replace\w*|stands\s+substituted)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.REPLACES, "MODIFIED"
        elif re.search(r"\b(amend\w*|modif\w*|stands\s+modified)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.MODIFIES, "MODIFIED"
        elif re.search(r"\b(supersed\w*|in\s+supersession)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.SUPERSEDES, "MODIFIED"
        elif re.search(r"\b(delete\w*|omit\w*|repeal\w*|rescind\w*)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.DELETES, "DELETED"
        elif re.search(r"\b(clarif\w*|guidance|faq|explained)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.CLARIFIES, "ACTIVE"
        elif re.search(r"\b(extend\w*)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.EXTENDS, "ACTIVE"
        elif re.search(r"\b(subject\s+to|conditional)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.SUBJECT_TO, "ACTIVE"
        elif re.search(r"\b(credited\s+to|credit\w*\s+to)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.CREDITED_TO, "MANDATORY"
        elif re.search(r"\b(payment\s+deadline|payable\s+within|payment\s+time)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.PAYMENT_DEADLINE, "MANDATORY"
        elif re.search(r"\b(submits?\s+to|forward\w*\s+to|send\w*\s+to)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.SUBMITS_TO, "MANDATORY"
        elif re.search(r"\b(reports?\s+to|disclos\w*\s+to)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.REPORTS_TO, "MANDATORY"
        elif re.search(r"\b(maintains?|preserve\w*|keep\w*)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.MAINTAINS, "MANDATORY"
        elif re.search(r"\b(implements?|execute\w*)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.IMPLEMENTS, "MANDATORY"
        elif re.search(r"\b(applies\s+to|applicable\s+to|govern\w*)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.APPLIES_TO, "ACTIVE"
        elif re.search(r"\b(refer\w*|in\s+terms\s+of|pursuant\s+to|cited)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.REFERENCES, "ACTIVE"
        elif re.search(r"\b(define\w*|means|deemed|understood\s+as)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.DEFINES, "ACTIVE"
        elif re.search(r"\b(shall|must|require\w*|obligat\w*|direct\w*|mandat\w*|instruct\w*|prescrib\w*|stipulat\w*|compl\w*|adhere\w*|submit\w*|maintain\w*|ensure\w*|provide\w*|bound\s+to|responsible\s+for|call\s+upon|enforce\w*)\b", clean_pred, re.IGNORECASE):
            return FactPredicate.REQUIRES, "MANDATORY"
        else:
            return None, None

    def _validate_and_enrich_facts(
        self,
        raw_facts: List[Any],
        document_id: str,
        chunk: Chunk,
        metadata: Optional[DocumentMetadata],
        model_name: str,
        require_grounded_evidence: bool = False,
    ) -> List[Fact]:
        """Validate atomic facts, reject empty/placeholder entries, and enrich with Python metadata."""
        valid_facts: List[Fact] = []
        chunk_text = chunk.content
        norm_chunk = normalize_text(chunk_text)

        PLACEHOLDERS = {
            "string", "...", "n/a", "na", "null", "none", "placeholder", "undefined",
            "[...]", "?", "<subject>", "<predicate>", "<object>", "tbd"
        }

        # Extract rule and clause anchors once per chunk
        rule_ref = None
        rule_match = re.search(r"\b(Rule\s*8\s*\([0-9a-zA-Z\(\)]+\)|Rule\s*\d+[a-zA-Z\(\)]*)", chunk_text, re.IGNORECASE)
        if rule_match:
            rule_ref = re.sub(r"\s+", " ", rule_match.group(1)).strip()

        clause_ref = None
        pt_match = re.search(r"\b(?:Point(?:\s*No\.?)?|Clause)\s*(\d+[a-zA-Z]?)", chunk_text, re.IGNORECASE)
        if pt_match:
            clause_ref = f"Point {pt_match.group(1)}"
        else:
            num_match = re.search(r"^\s*(\d+[a-zA-Z]?)\.\s+", chunk_text)
            if num_match:
                clause_ref = f"Point {num_match.group(1)}"
            else:
                q_match = re.search(r"\b(?:Question|FAQ)\s*(\d+)", chunk_text, re.IGNORECASE)
                if q_match:
                    clause_ref = f"Question {q_match.group(1)}"

        qualifiers = extract_qualifier_spans(chunk_text)

        for rf in raw_facts:
            if not isinstance(rf, dict):
                continue

            subj_raw = str(rf.get("subject") or "").strip()
            pred_raw = str(rf.get("predicate") or "").strip()
            obj_raw = str(rf.get("object") or "").strip()

            # Reject if any atomic triple element is empty
            if not subj_raw or not pred_raw or not obj_raw:
                logger.warning("Rejected fact: empty element in triple (subject=%r, predicate=%r, object=%r)", subj_raw, pred_raw, obj_raw)
                continue

            # Reject obvious hallucinated placeholders
            if (
                subj_raw.lower() in PLACEHOLDERS
                or pred_raw.lower() in PLACEHOLDERS
                or obj_raw.lower() in PLACEHOLDERS
            ):
                logger.warning("Rejected fact: placeholder detected (subject=%r, predicate=%r, object=%r)", subj_raw, pred_raw, obj_raw)
                continue

            # Reject unreasonable length or all-identical tokens
            if len(f"{subj_raw} {pred_raw} {obj_raw}") < 8 or (subj_raw.lower() == pred_raw.lower() == obj_raw.lower()):
                logger.warning("Rejected fact: invalid content length or identical triple tokens")
                continue

            predicate, reg_status = self._map_predicate_to_enum(pred_raw)
            if predicate is None:
                logger.warning("Rejected fact: invalid_predicate\npredicate=%r", pred_raw)
                continue

            # Ground source_text for provenance
            explicit_evidence = (rf.get("evidence") or rf.get("source_text") or "").strip()
            if require_grounded_evidence:
                if not explicit_evidence:
                    logger.warning("Rejected fact: empty_evidence\nevidence=%r", explicit_evidence)
                    continue
                if normalize_text(explicit_evidence) not in norm_chunk:
                    logger.warning("Rejected fact: evidence_not_found\nevidence=%r", explicit_evidence)
                    continue
                source_text = explicit_evidence
            else:
                if explicit_evidence and normalize_text(explicit_evidence) in norm_chunk:
                    source_text = explicit_evidence
                elif explicit_evidence and normalize_text(explicit_evidence) not in norm_chunk:
                    logger.warning("Rejected fact: evidence_not_found\nevidence=%r", explicit_evidence)
                    continue
                else:
                    source_text = self._find_grounded_sentence(chunk_text, subj_raw, obj_raw)

            raw_cond = rf.get("condition")
            conds = [raw_cond] if isinstance(raw_cond, str) and raw_cond.strip() else list(rf.get("conditions") or qualifiers.get("conditions") or [])

            raw_exc = rf.get("exception")
            excs = [raw_exc] if isinstance(raw_exc, str) and raw_exc.strip() else list(rf.get("exceptions") or qualifiers.get("exceptions") or [])

            threshold_val = rf.get("threshold") or qualifiers.get("threshold")
            time_val = rf.get("time") or qualifiers.get("time")
            source_circ_val = rf.get("source_circular") or qualifiers.get("source_circular")

            fact = Fact(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                hierarchy_node_id=chunk.parent_id,
                page_number=chunk.page_start,
                subject=subj_raw,
                predicate=predicate,
                object=obj_raw,
                rule_reference=rf.get("rule_reference") or rule_ref,
                clause_reference=rf.get("clause_reference") or clause_ref,
                conditions=conds,
                exceptions=excs,
                threshold=str(threshold_val) if threshold_val else None,
                time=str(time_val) if time_val else None,
                source_circular=str(source_circ_val) if source_circ_val else None,
                regulatory_status=rf.get("regulatory_status") or reg_status,
                effective_date=rf.get("effective_date"),
                confidence=float(rf.get("confidence", 0.95)),
                source_text=source_text,
                extraction_method="groq",
                model=model_name,
                status=FactStatus.ACTIVE,
                valid_from=metadata.document_date if metadata else None,
            )
            valid_facts.append(fact)

        return valid_facts

    async def _verify_facts(
        self,
        chunk: Chunk,
        facts: List[Fact],
        metadata: Optional[DocumentMetadata],
    ) -> List[Fact]:
        """Optional second-pass LLM audit: checks SUPPORTED, UNSUPPORTED, MODIFY, MISSING_FACT."""
        if not facts:
            return []

        facts_payload = [
            {
                "fact_index": i,
                "subject": f.subject,
                "predicate": f.predicate.value,
                "object": f.object,
            }
            for i, f in enumerate(facts)
        ]

        try:
            data = await self._generate_json(
                system_prompt=FACT_VERIFICATION_SYSTEM_PROMPT,
                user_prompt=FACT_VERIFICATION_USER_PROMPT.format(
                    chunk_text=chunk.content,
                    facts_json=json.dumps(facts_payload, indent=2),
                ),
                max_output_tokens=self.reserved_output_tokens,
                context_info=f"fact verification ({chunk.chunk_id})",
                temperature=0.0,
            )

            verified_facts: List[Fact] = []
            verifications = data.get("verifications", [])
            verdict_by_idx = {}
            for v in verifications:
                idx = v.get("fact_index")
                if idx is not None:
                    verdict_by_idx[idx] = v

            for i, fact in enumerate(facts):
                v_info = verdict_by_idx.get(i)
                if not v_info:
                    verified_facts.append(fact)
                    continue

                verdict = str(v_info.get("verdict", "")).upper()
                if verdict == "SUPPORTED":
                    verified_facts.append(fact)
                elif verdict == "MODIFY":
                    mod = v_info.get("modified_fact", {})
                    if mod.get("subject"):
                        fact.subject = mod["subject"]
                    if mod.get("predicate"):
                        new_pred, new_status = self._map_predicate_to_enum(mod["predicate"])
                        fact.predicate = new_pred
                        fact.regulatory_status = new_status
                    if mod.get("object"):
                        fact.object = mod["object"]
                    verified_facts.append(fact)
                elif verdict == "UNSUPPORTED":
                    logger.warning(
                        "Verifier rejected unsupported fact for chunk %s: [%s] %s %s (reason: %s)",
                        chunk.chunk_id,
                        fact.subject,
                        fact.predicate.value,
                        fact.object,
                        v_info.get("reason", "Not supported by text"),
                    )
                else:
                    verified_facts.append(fact)

            # Recover missing facts identified by verifier
            missing = data.get("missing_facts", [])
            if missing and isinstance(missing, list):
                enriched_missing = self._validate_and_enrich_facts(
                    raw_facts=missing,
                    document_id=chunk.document_id,
                    chunk=chunk,
                    metadata=metadata,
                    model_name=self.model,
                )
                for mf in enriched_missing:
                    logger.info(
                        "Verifier recovered missing fact for chunk %s: [%s] %s %s",
                        chunk.chunk_id, mf.subject, mf.predicate.value, mf.object
                    )
                    verified_facts.append(mf)

            return verified_facts

        except Exception as e:
            logger.warning("Second-pass verification encountered error for chunk %s: %s. Promoting candidate facts.", chunk.chunk_id, e)
            return facts

    async def _extract_chunk_with_cascade(
        self,
        document_id: str,
        chunk: Chunk,
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
        """Execute the accuracy-first fact extraction cascade for a single chunk:
        Stage 1: Normal extraction (temperature=0, max_tokens=1200, ACCURACY_FIRST prompt)
        Stage 2: Retry same chunk with same schema (on transient failure)
        Stage 3: Simplified extraction prompt (on json_validate_failed)
        Stage 4: Higher output-token budget (max_tokens=2000, on truncated output)
        Stage 5: Alternate/stronger extraction model (e.g. qwen/qwen3.8-27b)
        Stage 6: Terminal dead-letter marked EXTRACTION_FAILED (zero rule-based fallback into graph)
        """
        cid = chunk.chunk_id
        sanitized_text = sanitize_chunk_text(chunk.content)
        if not sanitized_text:
            return []

        attempt_count = 0
        cascade_history: List[str] = []
        last_error_code = 400
        last_error_msg = ""
        last_failed_gen = ""

        # Construct cascade stages
        stages = [
            (
                "stage_1_normal",
                self.model,
                self.reserved_output_tokens,
                ACCURACY_FIRST_FACT_EXTRACTION_SYSTEM_PROMPT,
                ACCURACY_FIRST_FACT_EXTRACTION_USER_PROMPT.format(chunk_text=sanitized_text),
            ),
            (
                "stage_2_retry_same",
                self.model,
                self.reserved_output_tokens,
                ACCURACY_FIRST_FACT_EXTRACTION_SYSTEM_PROMPT,
                ACCURACY_FIRST_FACT_EXTRACTION_USER_PROMPT.format(chunk_text=sanitized_text),
            ),
            (
                "stage_3_simplified_prompt",
                self.model,
                self.reserved_output_tokens,
                SIMPLIFIED_FALLBACK_FACT_EXTRACTION_SYSTEM_PROMPT,
                SIMPLIFIED_FALLBACK_FACT_EXTRACTION_USER_PROMPT.format(chunk_text=sanitized_text),
            ),
            (
                "stage_4_higher_token_budget",
                self.model,
                self.max_output_tokens_retry,
                SIMPLIFIED_FALLBACK_FACT_EXTRACTION_SYSTEM_PROMPT,
                SIMPLIFIED_FALLBACK_FACT_EXTRACTION_USER_PROMPT.format(chunk_text=sanitized_text),
            ),
        ]

        if self.alternate_model and self.alternate_model != self.model:
            stages.append((
                "stage_5_alternate_model",
                self.alternate_model,
                self.max_output_tokens_retry,
                SIMPLIFIED_FALLBACK_FACT_EXTRACTION_SYSTEM_PROMPT,
                SIMPLIFIED_FALLBACK_FACT_EXTRACTION_USER_PROMPT.format(chunk_text=sanitized_text),
            ))

        for stage_name, stage_model, stage_max_tokens, sys_prompt, usr_prompt in stages:
            attempt_count += 1
            cascade_history.append(f"{stage_name}:{stage_model}")
            logger.info(
                "Executing fact extraction cascade | stage=%s | chunk_id=%s | model=%s | max_tokens=%d | attempt=%d/%d",
                stage_name,
                cid,
                stage_model,
                stage_max_tokens,
                attempt_count,
                len(stages),
            )

            try:
                data, usage = await self._generate_json(
                    system_prompt=sys_prompt,
                    user_prompt=usr_prompt,
                    max_output_tokens=stage_max_tokens,
                    context_info=f"{stage_name} ({cid})",
                    model=stage_model,
                    temperature=0.0,
                    return_usage=True,
                )

                if isinstance(data, dict) and ("facts" in data or "results" in data):
                    raw_facts = data.get("facts", [])
                    if not raw_facts and "results" in data:
                        for r in data.get("results", []):
                            raw_facts.extend(r.get("facts", []))

                    valid_facts = self._validate_and_enrich_facts(
                        raw_facts=raw_facts,
                        document_id=document_id,
                        chunk=chunk,
                        metadata=metadata,
                        model_name=stage_model,
                    )

                    val_status = "VALID" if (valid_facts or len(raw_facts) == 0) else "EMPTY_REJECTED"
                    logger.info(
                        "Fact extraction telemetry | chunk_id=%s | facts_extracted=%d | validation_status=%s | "
                        "retry_count=%d | model=%s | input_tokens=%d | output_tokens=%d | stage=%s",
                        cid,
                        len(valid_facts),
                        val_status,
                        attempt_count - 1,
                        stage_model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        stage_name,
                    )

                    if self.verify_second_pass and valid_facts:
                        logger.info("Running second-pass verification for chunk %s (%d candidate facts)", cid, len(valid_facts))
                        valid_facts = await self._verify_facts(chunk, valid_facts, metadata)
                        logger.info("Second-pass verification completed for chunk %s (%d verified facts)", cid, len(valid_facts))

                    return valid_facts

            except GroqOutputTruncatedError as ote:
                last_error_code = 400
                last_error_msg = str(ote)
                last_failed_gen = ote.failed_generation
                logger.warning("Fact extraction output truncated | stage=%s | chunk_id=%s | %s", stage_name, cid, ote)
                continue

            except GroqJsonValidateError as jve:
                last_error_code = 400
                last_error_msg = str(jve)
                last_failed_gen = jve.failed_generation
                logger.warning("Fact extraction JSON validation failed | stage=%s | chunk_id=%s | %s", stage_name, cid, jve)
                continue

            except GroqRateLimitLongWaitError as rle:
                last_error_code = 429
                last_error_msg = f"Rate limit wait exceeded: {rle.retry_after:.1f}s"
                cooldown_sleep = min(rle.retry_after + 2.0, 300.0)
                logger.warning("Fact extraction received 429 long wait (%.1fs) | stage=%s | chunk_id=%s. Pausing cascade for %.1fs to allow rate limit to clear...", rle.retry_after, stage_name, cid, cooldown_sleep)
                await asyncio.sleep(cooldown_sleep)
                continue

            except GroqCircuitBreakerOpenError as cbe:
                last_error_code = 429
                last_error_msg = f"Circuit breaker open: {cbe.cooldown_remaining:.1f}s"
                breaker_sleep = cbe.cooldown_remaining + 1.0
                logger.warning("Fact extraction blocked by circuit breaker (%.1fs) | stage=%s | chunk_id=%s. Pausing cascade for %.1fs for breaker to reset...", cbe.cooldown_remaining, stage_name, cid, breaker_sleep)
                await asyncio.sleep(breaker_sleep)
                continue

            except Exception as e:
                last_error_code = getattr(getattr(e, "response", None), "status_code", 400)
                last_error_msg = str(e)
                logger.warning("Fact extraction exception | stage=%s | chunk_id=%s | %s", stage_name, cid, e)
                continue

        # Terminal Dead-Letter Stage (Stage 6)
        logger.error(
            "EXTRACTION_FAILED: All %d recovery cascade stages failed for chunk %s (document %s). "
            "No rule-based fallback will be inserted into knowledge graph.",
            attempt_count,
            cid,
            document_id,
        )

        if on_dead_letter:
            on_dead_letter(
                last_error_code,
                f"Fact extraction failed for chunk {cid} after {attempt_count} cascade attempts. Last error: {last_error_msg}",
                {
                    "provider": "groq",
                    "status": "EXTRACTION_FAILED",
                    "document_id": document_id,
                    "chunk_id": cid,
                    "error_code": last_error_code,
                    "error_message": last_error_msg,
                    "failed_generation": last_failed_gen[:300] if last_failed_gen else "",
                    "number_of_attempts": attempt_count,
                    "models_tried": list({s[1] for s in stages}),
                    "cascade_history": cascade_history,
                    "char_count": len(sanitized_text),
                    "preview": sanitized_text[:200],
                    "fallback_used": False,
                },
            )

        return []

    def _parse_raw_facts(
        self,
        data: Dict[str, Any],
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata],
    ) -> List[Fact]:
        """Parse raw facts from batch response, validating evidence and resolving chunk mapping."""
        raw_facts = data.get("facts", [])
        if not raw_facts and "results" in data:
            for r in data.get("results", []):
                raw_facts.extend(r.get("facts", []))

        facts: List[Fact] = []
        chunk_map = {c.chunk_id: c for c in chunks}

        for rf in raw_facts:
            try:
                cid = rf.get("chunk_id")
                matched_chunk = chunk_map.get(cid) if cid else (chunks[0] if chunks else None)
                if not matched_chunk:
                    continue

                enriched = self._validate_and_enrich_facts(
                    raw_facts=[rf],
                    document_id=document_id,
                    chunk=matched_chunk,
                    metadata=metadata,
                    model_name=self.model,
                    require_grounded_evidence=True,
                )
                facts.extend(enriched)
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
        """Extract atomic regulatory facts with content sanitization, single-chunk accuracy-first processing, and adaptive batch degradation."""
        if not chunks:
            return []

        # Content sanitization
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

        # 1. Default Accuracy-First: Single chunk processing
        if len(sanitized_chunks) == 1:
            return await self._extract_chunk_with_cascade(
                document_id=document_id,
                chunk=sanitized_chunks[0],
                metadata=metadata,
                on_dead_letter=on_dead_letter,
            )

        # 2. Multi-chunk batch handling (Adaptive Batch Splitting down to single chunks)
        chunk_ids = [c.chunk_id for c in sanitized_chunks]
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

        try:
            data = await self._generate_json(
                system_prompt=BATCH_FACT_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=self.reserved_output_tokens,
                context_info=f"batch of {len(sanitized_chunks)} chunks ({','.join(chunk_ids)})",
            )
            if data and ("facts" in data or "results" in data):
                facts = self._parse_raw_facts(data, document_id, sanitized_chunks, metadata)
                logger.info("Successfully extracted %d facts via Groq for batch of %d chunks", len(facts), len(sanitized_chunks))
                return facts
        except Exception as batch_err:
            logger.warning("Batch fact extraction encountered %s. Splitting batch into single chunks for accuracy...", batch_err)

        # Split batch recursively until single chunks are processed with the accuracy cascade
        mid = len(sanitized_chunks) // 2
        facts_first = await self.extract_facts_batch(document_id, sanitized_chunks[:mid], metadata, on_dead_letter)
        facts_second = await self.extract_facts_batch(document_id, sanitized_chunks[mid:], metadata, on_dead_letter)
        return facts_first + facts_second

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
        """Deterministic rule-based fallback extracting complete regulatory clauses and rejecting metadata."""
        facts: List[Fact] = []
        for chunk in chunks:
            text = chunk.content.strip()
            if len(text) < 30:
                continue

            # Skip pure metadata or header chunks
            if re.search(r"^(?:NATIONAL\s+STOCK|CIRCULAR\s+Ref|Download\s+Ref|DEPARTMENT:|Date:|To\s+All\s+Members|Yours\s+faithfully|CIN\s*:|Page\s+\d+)", text, re.IGNORECASE):
                continue

            # Extract distinct numbered points or statutory provisos
            clause_blocks = re.split(r"\n(?=(?:\d+\.|\([a-z0-9]+\)|Point(?:\s*No\.?)?\s*\d+|“Provided|Question\s*\d+:))", text)

            for block in clause_blocks:
                block_clean = block.strip()
                if len(block_clean) < 25:
                    continue

                if re.search(r"^(?:Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page \d+|CIN|Tel No|To All Members|Yours faithfully)", block_clean, re.IGNORECASE):
                    continue

                # Determine predicate deterministically without defaulting to REQUIRES
                predicate = None
                regulatory_status = "ACTIVE"

                if re.search(r"\b(shall\s+be\s+replaced|shall\s+replace|replaces?|shall\s+be\s+substituted|substitut\w*)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.REPLACES
                    regulatory_status = "MODIFIED"
                elif re.search(r"\b(supersedes?|in\s+supersession\s+of)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.SUPERSEDES
                    regulatory_status = "MODIFIED"
                elif re.search(r"\b(repealed?|repeals?|rescinded?)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.REPEALS
                    regulatory_status = "DELETED"
                elif re.search(r"\b(deleted?|deletes?|omitted?|stand\s+deleted)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.DELETES
                    regulatory_status = "DELETED"
                elif re.search(r"\b(shall\s+be\s+modified|amended?|amends?|modifi\w+|revised?)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.MODIFIES
                    regulatory_status = "MODIFIED"
                elif re.search(r"\b(prohibit\w*|shall\s+not|must\s+not|cannot|forbidden|barred|non-compliance)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.PROHIBITS
                    regulatory_status = "PROHIBITED"
                elif re.search(r"\b(permit\w*|allowed|may|authorized|eligible|not\s+be\s+construed\s+as\s+business)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.PERMITS
                    regulatory_status = "PERMITTED"
                elif re.search(r"\b(clarif\w+|guidance|faq|explained)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.CLARIFIES
                elif re.search(r"\b(defines?|means|shall\s+be\s+understood)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.DEFINES
                elif re.search(r"\b(reference\s+is\s+invited\s+to|refer(?:s|red)?\s+to|circular\s+no\.)\b", block_clean, re.IGNORECASE):
                    # Citations are reference facts, NEVER amendments or requirements
                    predicate = FactPredicate.REFERENCES
                elif re.search(r"\b(shall|must|required?|mandat\w*|direct\w*|instruct\w*|prescrib\w*|stipulat\w*|maintain|submit|ensure|comply|complies)\b", block_clean, re.IGNORECASE):
                    predicate = FactPredicate.REQUIRES
                    regulatory_status = "MANDATORY"
                else:
                    # If no valid regulatory proposition can be established, do not create fact
                    continue

                subject = "Trading Member"
                if re.search(r"\b(Asset\s+Management\s+Company|Asset\s+Management\s+Companies|AMC|AMCs)\b", block_clean, re.IGNORECASE):
                    subject = "Asset Management Company"
                elif re.search(r"\b(Mutual\s+Fund|Mutual\s+Funds)\b", block_clean, re.IGNORECASE):
                    subject = "Mutual Fund"
                elif re.search(r"\b(Exchange|NSE|SEBI|Board|Regulator)\b", block_clean, re.IGNORECASE):
                    subject = "Exchange / Regulator"
                elif re.search(r"\b(Client|Investor)\b", block_clean, re.IGNORECASE):
                    subject = "Client / Investor"

                # Extract anchors
                rule_ref = None
                rule_match = re.search(r"\b(Rule\s*8\s*\([0-9a-zA-Z\(\)]+\)|Rule\s*\d+[a-zA-Z\(\)]*)", block_clean, re.IGNORECASE)
                if rule_match:
                    rule_ref = re.sub(r"\s+", " ", rule_match.group(1)).strip()

                clause_ref = None
                pt_match = re.search(r"\b(?:Point(?:\s*No\.?)?|Clause)\s*(\d+[a-zA-Z]?)", block_clean, re.IGNORECASE)
                if pt_match:
                    clause_ref = f"Point {pt_match.group(1)}"
                else:
                    num_match = re.search(r"^\s*(\d+[a-zA-Z]?)\.\s+", block_clean)
                    if num_match:
                        clause_ref = f"Point {num_match.group(1)}"
                    else:
                        q_match = re.search(r"\b(?:Question|FAQ)\s*(\d+)", block_clean, re.IGNORECASE)
                        if q_match:
                            clause_ref = f"Question {q_match.group(1)}"

                # Clean object text
                first_line = block_clean.split("\n")[0]
                obj_text = first_line[:150]

                # Deterministically preserve conditions and exceptions
                qualifiers = extract_qualifier_spans(block_clean)

                fact = Fact(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    hierarchy_node_id=chunk.parent_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj_text,
                    rule_reference=rule_ref,
                    clause_reference=clause_ref,
                    conditions=qualifiers.get("conditions", []),
                    exceptions=qualifiers.get("exceptions", []),
                    threshold=qualifiers.get("threshold"),
                    time=qualifiers.get("time"),
                    source_circular=qualifiers.get("source_circular"),
                    regulatory_status=regulatory_status,
                    confidence=0.75,
                    source_text=block_clean[:350],
                    extraction_method="rule_based",
                    model=None,
                    fallback_reason="groq_fallback",
                    status=FactStatus.ACTIVE,
                    valid_from=metadata.document_date if metadata else None,
                )
                facts.append(fact)

        return facts

