import pytest
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

def test_rag_endpoint(api_key):
    """
    Тестирование эндпоинта /api/rag/ для генерации ответа с использованием RAG.
    """
    headers = {"api-key": api_key}

    # Входной запрос пользователя
    user_query = {"user_query": "Как управлять освещением в проекте?"}

    # Мокирование поиска по базе знаний
    mock_search_results = [
        {"id": 1, "content": "Управление освещением через мобильное приложение."},
        {"id": 2, "content": "Система позволяет регулировать яркость и включать/выключать свет."}
    ]

    # Мокирование ответа от LLM
    mock_llm_response = "В проекте управление освещением реализовано через мобильное приложение. Пользователи могут регулировать яркость и включать/выключать свет."

    with mock.patch("services.indexing_service.indexing_service.search_documents", return_value=mock_search_results), \
         mock.patch("services.openai_service.OpenAIService.generate_response", return_value=mock_llm_response):

        response = client.post("/api/rag/", headers=headers, json=user_query)

        assert response.status_code == 200
        response_data = response.json()

        # Проверяем, что ответ содержит сгенерированный текст и источники
        assert "response" in response_data
        assert response_data["response"] == mock_llm_response
        assert "sources" in response_data
        assert len(response_data["sources"]) == len(mock_search_results)