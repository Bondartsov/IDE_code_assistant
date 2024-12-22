# tests/test_openai_service.py

import pytest
from services.openai_service import OpenAIService
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
import openai
from openai.error import OpenAIError

@pytest.mark.asyncio
async def test_generate_response_openai():
    """
    Тест генерации ответа через OpenAI API.
    """
    openai_service = OpenAIService()

    with patch('openai.ChatCompletion.acreate') as mock_acreate:
        mock_acreate.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    content="Test response"
                )
            )]
        )

        response = await openai_service.generate_response("Test prompt")
        assert response == "Test response"

@pytest.mark.asyncio
async def test_generate_response_openai_error():
    """
    Тест обработки ошибки при вызове OpenAI API.
    """
    openai_service = OpenAIService()

    with patch('openai.ChatCompletion.acreate', side_effect=OpenAIError("Test error")):
        with pytest.raises(HTTPException) as exc_info:
            await openai_service.generate_response("Test prompt")
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "OpenAI API error"