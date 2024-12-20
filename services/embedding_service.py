# Файл: services/embedding_service.py

import numpy as np
import openai
from fastapi import HTTPException
import tiktoken
import requests

from core.config import settings
from core.logger import logger

def num_tokens(text: str, model_name: str = "text-embedding-ada-002") -> int:
    """
    Подсчет количества токенов в тексте для указанной модели.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_text(text: str, max_tokens: int, model_name: str = "text-embedding-ada-002"):
    """
    Разбиение текста на части с определенным количеством токенов.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def generate_embedding(text: str):
    """
    Генерация эмбеддинга для текста.
    """
    api_provider = settings.API_PROVIDER

    if api_provider == "openai":
        try:
            model_name = "text-embedding-ada-002"
            response = openai.Embedding.create(
                input=text,
                model=model_name
            )
            embedding = response["data"][0]["embedding"]
            return embedding
        except openai.error.OpenAIError as e:
            logger.error(f"Ошибка генерации эмбеддинга: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")
    elif api_provider == "lmstudio":
        # Если LMStudio не поддерживает эмбеддинги, возвращаем ошибку
        logger.error("Эмбеддинги не поддерживаются выбранным провайдером API")
        raise HTTPException(status_code=501, detail="Embeddings are not supported with the selected API provider.")
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")