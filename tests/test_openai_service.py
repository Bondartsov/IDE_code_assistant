# tests/test_openai_service.py

import pytest
from services.openai_service import generate_response, get_models
from core.config import settings
from unittest.mock import MagicMock
from fastapi import HTTPException
import openai

@pytest.mark.skipif(settings.API_PROVIDER != "openai", reason="Тест предназначен для OpenAI")
def test_generate_response_openai(mocker):
    # Создаём мок-объекты
    mock_message = MagicMock()
    mock_message.content = "Test response"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    # Мокируем openai.ChatCompletion.create
    mocker.patch('openai.ChatCompletion.create', return_value=mock_completion)

    response = generate_response("Test prompt")
    assert response == "Test response"

@pytest.mark.skipif(settings.API_PROVIDER != "openai", reason="Тест предназначен для OpenAI")
def test_generate_response_openai_error(mocker):
    # Мокируем openai.ChatCompletion.create, чтобы он вызывал исключение
    mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.OpenAIError("Test error"))

    with pytest.raises(HTTPException) as exc_info:
        generate_response("Test prompt")
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "OpenAI API error"