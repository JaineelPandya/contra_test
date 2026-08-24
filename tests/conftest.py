"""Pytest fixtures and test configuration."""
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from app.core.config import get_settings
from app.llm.mock_provider import MockProvider
from app.main import app
from tests.generate_sample_pdfs import generate_2022_circular, generate_2026_circular


@pytest.fixture(scope="session")
def sample_pdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate temporary synthetic PDFs for test session."""
    tmp_dir = tmp_path_factory.mktemp("sample_pdfs")
    generate_2022_circular(tmp_dir / "circular_2022_50957.pdf")
    generate_2026_circular(tmp_dir / "circular_2026_amendment.pdf")
    return tmp_dir


@pytest.fixture
def mock_llm_provider() -> MockProvider:
    """Return a deterministic mock LLM provider instance."""
    return MockProvider()


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient fixture."""
    with TestClient(app) as client:
        yield client
