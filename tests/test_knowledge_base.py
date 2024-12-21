# tests/test_knowledge_base.py

import pytest
from fastapi.testclient import TestClient
from main import app
from unittest import mock
from services.indexing_service import index_manager

client = TestClient(app)

def get_api_key():
    response = client.post("/api/generate_key/")
    assert response.status_code == 200
    return response.json()["api_key"]

@pytest.fixture
def api_key():
    key = get_api_key()
    yield key
    # Отзываем ключ после тестов
    client.post(f"/api/expire_key/{key}/")

def test_add_document(api_key):
    headers = {"api-key": api_key}
    data = {
        "title": "Test Document",
        "content": "This is a test content."
    }
    with mock.patch('services.embedding_service.generate_embedding', return_value=[0.0]*1536), \
         mock.patch.object(index_manager, 'add_document', return_value=None):

        response = client.post("/api/knowledge_base/", headers=headers, json=data)
        assert response.status_code == 200
        assert response.json()["detail"] == "Document added successfully"
        assert "document_id" in response.json()
