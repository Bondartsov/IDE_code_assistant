# tests/test_knowledge_base.py

import pytest
from io import BytesIO
from unittest import mock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def api_key():
    """
    Фикстура для генерации API-ключа.
    """
    response = client.post("/api/generate_key/", json={"name": "Test Application"})
    assert response.status_code == 200
    return response.json()["api_key"]

def test_add_document(api_key):
    """
    Тестирование добавления документа в базу знаний.
    """
    headers = {"api-key": api_key}

    # Подготовка файлов для отправки
    files = [
        (
            "files",
            (
                "test.txt",
                BytesIO(b"This is a test content."),
                "text/plain"
            ),
        ),
    ]

    # Подготовка текстовых данных
    data = {
        "texts": "This is additional text data."
    }

    with mock.patch('services.embedding_service.generate_embedding', return_value=[0.0]*1536), \
         mock.patch('services.indexing_service.index_manager.add_document', return_value=None):

        response = client.post(
            "/api/knowledge_base/",
            headers=headers,
            files=files,
            data=data
        )

        print("Response status code:", response.status_code)
        print("Response JSON:", response.json())
        assert response.status_code == 200