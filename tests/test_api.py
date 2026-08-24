"""FastAPI endpoints integration tests."""
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from app.models.document import Document, ProcessingStatus, ProcessingStage


def test_health_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "neo4j" in data


def test_upload_invalid_file_type(test_client: TestClient) -> None:
    file_data = io.BytesIO(b"Hello world")
    response = test_client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", file_data, "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


@patch("app.api.documents.DocumentPipeline.process_document")
@patch("app.api.documents.GraphRepository")
def test_upload_pdf_success(mock_repo_class: MagicMock, mock_pipeline_proc: MagicMock, test_client: TestClient, sample_pdf_dir: Path) -> None:
    mock_repo = MagicMock()
    mock_repo_class.return_value = mock_repo

    pdf_path = sample_pdf_dir / "circular_2022_50957.pdf"
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    response = test_client.post(
        "/api/documents/upload",
        files={"file": ("circular_2022_50957.pdf", io.BytesIO(file_bytes), "application/pdf")},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["document_id"] is not None
    assert data["filename"] == "circular_2022_50957.pdf"
    assert data["status"] == "PROCESSING"
    assert mock_repo.save_document.called


@patch("app.api.processing.GraphRepository")
def test_get_status_endpoint(mock_repo_class: MagicMock, test_client: TestClient) -> None:
    mock_repo = MagicMock()
    mock_repo.get_document.return_value = Document(
        document_id="doc-test-uuid",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        status=ProcessingStatus.PROCESSING,
        current_stage=ProcessingStage.FACT_EXTRACTION,
        pages_processed=2,
        total_pages=2,
        sections_created=5,
        chunks_created=10,
        facts_extracted=18,
    )
    mock_repo_class.return_value = mock_repo

    response = test_client.get("/api/documents/doc-test-uuid/status")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-test-uuid"
    assert data["current_stage"] == "FACT_EXTRACTION"
    assert data["facts_extracted"] == 18


@patch("app.api.documents.GraphRepository")
def test_list_and_get_document_endpoints(mock_repo_class: MagicMock, test_client: TestClient) -> None:
    mock_repo = MagicMock()
    sample_doc = Document(
        document_id="doc-test-uuid",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        reference_number="NSE/COMP/50957",
        circular_number="02/2022",
        document_date="2022-01-07",
    )
    mock_repo.list_documents.return_value = [sample_doc]
    mock_repo.get_document.return_value = sample_doc
    mock_repo_class.return_value = mock_repo

    # Test list
    res_list = test_client.get("/api/documents")
    assert res_list.status_code == 200
    assert len(res_list.json()) == 1

    # Test get by ID
    res_get = test_client.get("/api/documents/doc-test-uuid")
    assert res_get.status_code == 200
    assert res_get.json()["reference_number"] == "NSE/COMP/50957"


@patch("app.api.documents.GraphRepository")
def test_delete_document_endpoint(mock_repo_class: MagicMock, test_client: TestClient) -> None:
    mock_repo = MagicMock()
    mock_repo.delete_document.return_value = True
    mock_repo_class.return_value = mock_repo

    response = test_client.delete("/api/documents/doc-test-uuid")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]
