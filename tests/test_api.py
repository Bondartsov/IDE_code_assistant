# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

def get_api_key():
    """
    Генерация и получение API-ключа для тестов.
    """
    response = client.post("/api/generate_key/", json={"name": "Test Application"})
    assert response.status_code == 200
    return response.json()["api_key"]

def test_generate_key():
    """
    Тест генерации нового API-ключа.
    """
    response = client.post("/api/generate_key/", json={"name": "Test Application"})
    assert response.status_code == 200
    assert "api_key" in response.json()

def test_get_models():
    """
    Тест получения списка моделей OpenAI или LMStudio.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}
    with patch('services.openai_service.OpenAIService.get_models', return_value=["model-a", "model-b"]):
        response = client.get("/api/models/", headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

def test_run_openai_prompt():
    """
    Тест выполнения запроса к эндпоинту /api/openai/.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}
    data = {
        "prompt": "What is the capital of France?"
    }
    with patch('services.openai_service.OpenAIService.generate_response', return_value="Paris"):
        response = client.post("/api/openai/", headers=headers, json=data)
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["response"] == "Paris"

def test_api_key_expiry_cleanup():
    """
    Тест отзыва и проверки недействительного API-ключа.
    """
    api_key = get_api_key()
    # Отзываем ключ
    from core.security import api_key_manager
    api_key_manager.invalidate_api_key(api_key)
    # Проверяем, что ключ больше недействителен
    headers = {"api-key": api_key}
    response = client.get("/api/models/", headers=headers)
    assert response.status_code == 401