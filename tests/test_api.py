"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from deployment.api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_caption_endpoint_no_file():
    """Test caption endpoint without file."""
    response = client.post("/caption")
    assert response.status_code != 200  # Should fail without file


# Note: Add more tests for actual image captioning once model is loaded

