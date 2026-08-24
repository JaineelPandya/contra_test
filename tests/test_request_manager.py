"""Unit tests for Centralized Gemini Request Manager."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from app.llm.request_manager import GeminiRequestManager, sanitize_url


def test_sanitize_url():
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=AIzaSySecret12345&other=val"
    safe = sanitize_url(url)
    assert "AIzaSySecret12345" not in safe
    assert "key=***REDACTED***" in safe


@pytest.mark.asyncio
async def test_retry_on_429_with_retry_after():
    manager = GeminiRequestManager()
    manager.max_retries = 3
    manager.initial_delay = 0.01

    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0"}

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {"candidates": [{"content": {"parts": [{"text": '{"status": "ok"}'}]}}]}

    with patch("httpx.AsyncClient.post", side_effect=[resp_429, resp_200]) as mock_post:
        res = await manager.execute_request(
            url="https://api.test?key=123",
            payload={"test": 1},
        )
        assert res == {"candidates": [{"content": {"parts": [{"text": '{"status": "ok"}'}]}}]}
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_503_temporary_failure():
    manager = GeminiRequestManager()
    manager.max_retries = 3
    manager.initial_delay = 0.01

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.headers = {}

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {"status": "recovered"}

    with patch("httpx.AsyncClient.post", side_effect=[resp_503, resp_200]) as mock_post:
        res = await manager.execute_request(
            url="https://api.test?key=123",
            payload={"test": 1},
        )
        assert res == {"status": "recovered"}
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_dead_letter_callback_on_exhausted_retries():
    manager = GeminiRequestManager()
    manager.max_retries = 2
    manager.initial_delay = 0.01

    resp_500 = MagicMock(spec=httpx.Response)
    resp_500.status_code = 500
    resp_500.headers = {}
    resp_500.raise_for_status.side_effect = httpx.HTTPStatusError("500 Internal Error", request=MagicMock(), response=resp_500)

    dead_letter_called = []

    def on_dl(code, msg):
        dead_letter_called.append((code, msg))

    with patch("httpx.AsyncClient.post", return_value=resp_500):
        with pytest.raises(httpx.HTTPStatusError):
            await manager.execute_request(
                url="https://api.test?key=123",
                payload={"test": 1},
                on_dead_letter=on_dl,
            )

    assert len(dead_letter_called) == 1
    assert dead_letter_called[0][0] == 500
