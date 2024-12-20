# Файл: tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def get_api_key():
    """
    Генерация и получение API-ключа для тестов.
    """
    response = client.post("/api/generate_key/")
    assert response.status_code == 200
    return response.json()["api_key"]

def test_generate_key():
    """
    Тест генерации нового API-ключа.
    """
    response = client.post("/api/generate_key/")
    assert response.status_code == 200
    assert "api_key" in response.json()

def test_get_models():
    """
    Тест получения списка моделей OpenAI или LMStudio.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}
    response = client.get("/api/models/", headers=headers)
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) > 0

def test_run_openai_prompt():
    """
    Тест выполнения запроса к эндпоинту /api/openai/.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}
    data = {
        "prompt": "What is the capital of France?"
    }
    response = client.post("/api/openai/", headers=headers, data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_api_key_expiry_cleanup():
    """
    Тест отзыва и проверки недействительного API-ключа.
    """
    api_key = get_api_key()
    # Отзываем ключ
    expire_response = client.post(f"/api/expire_key/{api_key}/")
    assert expire_response.status_code == 200
    # Проверяем, что ключ больше недействителен
    headers = {"api-key": api_key}
    response = client.get("/api/models/", headers=headers)
    assert response.status_code == 401

# Дополнительные тесты могут быть добавлены по необходимости