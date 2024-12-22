# tests/test_embedding_service.py

import pytest
from services.embedding_service import embedding_service
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.mark.asyncio
async def test_num_tokens():
    """
    Тест подсчёта количества токенов.
    """
    with patch('tiktoken.encoding_for_model') as mock_encoding_for_model:
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_encoding_for_model.return_value = mock_encoding

        text = "Hello, world!"
        tokens = embedding_service.num_tokens(text)
        assert tokens == 3

@pytest.mark.asyncio
async def test_split_text():
    """
    Тест разбиения текста на части по количеству токенов.
    """
    with patch('tiktoken.encoding_for_model') as mock_encoding_for_model:
        mock_encoding = MagicMock()
        # Создаём 1000 токенов
        mock_encoding.encode.return_value = list(range(1000))
        # Возвращаем строку 'chunk' при декодировании
        mock_encoding.decode.side_effect = lambda tokens: 'chunk'

        mock_encoding_for_model.return_value = mock_encoding

        text = "Hello, world! " * 1000  # Длинный текст
        max_tokens = 50
        chunks = embedding_service.split_text(text, max_tokens)
        assert len(chunks) == 20  # 1000 токенов разделить на чанки по 50 токенов

@pytest.mark.asyncio
async def test_generate_embedding_openai():
    """
    Тест генерации эмбеддинга с использованием OpenAI.
    """
    with patch('services.openai_service.OpenAIService.generate_embedding', return_value=[0.0]*1536):
        embedding = await embedding_service.generate_embedding("Test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536