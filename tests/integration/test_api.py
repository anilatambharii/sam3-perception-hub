"""Integration tests for perception API."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from sam3_perception.main import app
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_concepts_validate(client):
    response = client.get("/api/v1/concepts/validate?concept=person")
    assert response.status_code == 200
    data = response.json()
    assert "allowed" in data


@pytest.mark.slow
def test_segment_concept(client):
    # Would need actual image for full test
    pass
