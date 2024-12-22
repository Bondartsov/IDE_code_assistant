# tests/test_embedding_service.py

import pytest
from services.embedding_service import generate_embedding, num_tokens, split_text
from core.config import settings
from unittest.mock import patch

def test_num_tokens():
    """
    Тест подсчёта количества токенов.
    """
    text = "Hello, world!"
    tokens = num_tokens(text)
    assert tokens > 0

def test_split_text():
    """
    Тест разбиения текста на части по количеству токенов.
    """
    text = "Hello, world! " * 1000  # Длинный текст
    max_tokens = 50
    chunks = split_text(text, max_tokens)
    assert len(chunks) > 1

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