# tests/test_embedding_service.py

import pytest
from services.embedding_service import generate_embedding, num_tokens, split_text
from core.config import settings
from unittest import mock

def test_num_tokens():
    text = "Hello, world!"
    tokens = num_tokens(text)
    assert tokens > 0

def test_split_text():
    text = "Hello, world! " * 1000  # Длинный текст
    max_tokens = 50
    chunks = split_text(text, max_tokens)
    assert len(chunks) > 1

@pytest.mark.skipif(settings.API_PROVIDER != "openai", reason="Тест предназначен для OpenAI")
def test_generate_embedding_openai(mocker):
    # Мокируем openai.Embedding.create
    mocker.patch('openai.Embedding.create', return_value={
        'data': [{
            'embedding': [0.0]*1536
        }]
    })
    embedding = generate_embedding("Test text")
    assert len(embedding) == 1536