# services/embedding_service.py

"""
Модуль для генерации эмбеддингов текста.
"""

from typing import List
import numpy as np
import openai
from fastapi import HTTPException
import tiktoken
import requests

from core.config import settings
from core.logger import logger

def num_tokens(text: str, model_name: str = "text-embedding-ada-002") -> int:
    """
    Подсчитывает количество токенов в тексте для указанной модели.

    Args:
        text (str): Текст для подсчета токенов.
        model_name (str): Имя модели.

    Returns:
        int: Количество токенов.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def split_text(
    text: str, max_tokens: int, model_name: str = "text-embedding-ada-002"
) -> List[str]:
    """
    Разбивает текст на части по максимальному количеству токенов.

    Args:
        text (str): Исходный текст.
        max_tokens (int): Максимальное количество токенов в части.
        model_name (str): Имя модели.

    Returns:
        List[str]: Список частей текста.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    chunks = [
        encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)
    ]
    return chunks

def generate_embedding(text: str) -> List[float]:
    """
    Генерирует эмбеддинг для текста.

    Args:
        text (str): Текст для обработки.

    Returns:
        List[float]: Эмбеддинг текста.

    Raises:
        HTTPException: Ошибка при обращении к API провайдера.
    """
    api_provider = settings.API_PROVIDER

    if api_provider == "openai":
        try:
            model_name = "text-embedding-ada-002"
            openai.api_key = settings.OPENAI_API_KEY
            response = openai.Embedding.create(
                input=text,
                model=model_name
            )
            embedding = response["data"][0]["embedding"]
            return embedding
        except openai.error.OpenAIError as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")
    elif api_provider == "lmstudio":
        try:
            lmstudio_api_url = settings.LMSTUDIO_EMBEDDING_API_URL
            headers = {"Content-Type": "application/json"}
            payload = {"input": text}
            response = requests.post(
                lmstudio_api_url,
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                return embedding
            else:
                logger.error(f"Error from LMStudio API: {response.text}")
                raise HTTPException(status_code=500, detail="LMStudio API error")
        except Exception as e:
            logger.error(f"Error accessing LMStudio API: {e}")
            raise HTTPException(status_code=500, detail="LMStudio API error")
    else:
        logger.error("Unsupported API provider")
        raise HTTPException(status_code=500, detail="Unsupported API provider")