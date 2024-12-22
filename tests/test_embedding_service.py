# tests/test_embedding_service.py

import pytest
from services.embedding_service import generate_embedding, num_tokens, split_text
from core.config import settings
from unittest.mock import patch, MagicMock

def test_num_tokens():
    """
    Тест подсчёта количества токенов.
    """
    with patch('tiktoken.encoding_for_model') as mock_encoding_for_model:
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_encoding_for_model.return_value = mock_encoding

        text = "Hello, world!"
        tokens = num_tokens(text)
        assert tokens == 3

def test_split_text():
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
        chunks = split_text(text, max_tokens)
        assert len(chunks) == 20  # 1000 токенов разделить на чанки по 50 токенов

@pytest.mark.skipif(settings.API_PROVIDER != "openai", reason="Тест предназначен для OpenAI")
def test_generate_embedding_openai():
    """
    Тест генерации эмбеддинга с использованием OpenAI.
    """
    with patch('openai.Embedding.create') as mock_create:
        mock_create.return_value = {
            'data': [{
                'embedding': [0.0]*1536
            }]
        }
        embedding = generate_embedding("Test text")
        assert len(embedding) == 1536