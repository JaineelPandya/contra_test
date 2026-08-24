"""Centralized Gemini Request Manager with global concurrency, rate limiting, and backoff retries."""
import asyncio
from collections import deque
import random
import re
import time
from typing import Any, Callable, Dict, Optional
import httpx
from app.core.config import get_settings
from app.core.logging import logger

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def sanitize_url(url: str) -> str:
    """Redact sensitive API keys from URL query parameters for safe logging."""
    return re.sub(r"(key=)[^&\s]+", r"\1***REDACTED***", str(url))


class GeminiRequestManager:
    """Centralized manager controlling all Gemini API traffic with concurrency and rate limiting."""

    _instance: Optional["GeminiRequestManager"] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._semaphore = asyncio.Semaphore(self.settings.GEMINI_MAX_CONCURRENT_REQUESTS)
        self._lock = asyncio.Lock()
        self._request_timestamps: deque = deque()
        self.rpm_limit = self.settings.GEMINI_REQUESTS_PER_MINUTE
        self.max_retries = self.settings.GEMINI_MAX_RETRIES
        self.initial_delay = self.settings.GEMINI_INITIAL_RETRY_DELAY
        self.max_delay = self.settings.GEMINI_MAX_RETRY_DELAY

    @classmethod
    def get_instance(cls) -> "GeminiRequestManager":
        """Singleton accessor for global request manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _enforce_rate_limit(self) -> None:
        """Sliding window rate limiter enforcing requests per minute."""
        async with self._lock:
            now = time.monotonic()
            # Prune timestamps older than 60 seconds
            while self._request_timestamps and now - self._request_timestamps[0] > 60.0:
                self._request_timestamps.popleft()

            if len(self._request_timestamps) >= self.rpm_limit:
                oldest = self._request_timestamps[0]
                sleep_time = max(0.1, 60.0 - (now - oldest) + 0.1)
                logger.warning(
                    "Gemini RPM limit (%d) reached. Throttling request for %.2f seconds.",
                    self.rpm_limit,
                    sleep_time,
                )
                await asyncio.sleep(sleep_time)

            self._request_timestamps.append(time.monotonic())

    async def execute_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        on_dead_letter: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a Gemini POST request through global concurrency, rate limit, and exponential backoff retries."""
        safe_url = sanitize_url(url)
        logger.debug("Request submitted to Gemini Request Manager: %s", safe_url)

        async with self._semaphore:
            attempt = 0
            while True:
                attempt += 1
                await self._enforce_rate_limit()

                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(url, json=payload, headers=headers or {})

                        if response.status_code == 200:
                            logger.debug("Gemini request successful on attempt %d: %s", attempt, safe_url)
                            return response.json()

                        status_code = response.status_code

                        if status_code in RETRYABLE_STATUS_CODES and attempt <= self.max_retries:
                            # Extract Retry-After header if present
                            retry_after_hdr = response.headers.get("Retry-After")
                            if retry_after_hdr and retry_after_hdr.isdigit():
                                delay = float(retry_after_hdr)
                            else:
                                base_delay = min(self.max_delay, self.initial_delay * (2 ** (attempt - 1)))
                                jitter = random.uniform(0.1, 0.5 * base_delay)
                                delay = base_delay + jitter

                            if status_code == 429:
                                logger.warning(
                                    "Gemini 429 Too Many Requests (attempt %d/%d). Retrying in %.2f seconds...",
                                    attempt,
                                    self.max_retries,
                                    delay,
                                )
                            else:
                                logger.warning(
                                    "Gemini temporary server failure (HTTP %d, attempt %d/%d). Retrying in %.2f seconds...",
                                    status_code,
                                    attempt,
                                    self.max_retries,
                                    delay,
                                )

                            await asyncio.sleep(delay)
                            continue

                        # Non-retryable error or retries exhausted
                        response.raise_for_status()

                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if status_code in RETRYABLE_STATUS_CODES and attempt <= self.max_retries:
                        base_delay = min(self.max_delay, self.initial_delay * (2 ** (attempt - 1)))
                        delay = base_delay + random.uniform(0.1, 0.5 * base_delay)
                        logger.warning(
                            "Gemini HTTP error %d (attempt %d/%d). Retrying in %.2f seconds...",
                            status_code,
                            attempt,
                            self.max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.error(
                        "Gemini request retries exhausted or non-retryable error (HTTP %d): %s",
                        status_code,
                        safe_url,
                    )
                    if on_dead_letter:
                        on_dead_letter(status_code, str(exc))
                    raise

                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                    if attempt <= self.max_retries:
                        base_delay = min(self.max_delay, self.initial_delay * (2 ** (attempt - 1)))
                        delay = base_delay + random.uniform(0.1, 0.5 * base_delay)
                        logger.warning(
                            "Gemini network connection/timeout error '%s' (attempt %d/%d). Retrying in %.2f seconds...",
                            type(exc).__name__,
                            attempt,
                            self.max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.error("Gemini network error retries exhausted: %s", exc)
                    if on_dead_letter:
                        on_dead_letter(503, str(exc))
                    raise


# Global manager singleton
gemini_request_manager = GeminiRequestManager.get_instance()
