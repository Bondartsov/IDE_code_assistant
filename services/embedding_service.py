# services/embedding_service.py

import asyncio
from typing import List, Optional
import numpy as np
from core.config import settings
from services.openai_service import OpenAIService
from services.lmstudio_service import LMStudioService
from core.logger import logger

# Попытка импортировать tiktoken
try:
    import tiktoken
except ImportError:
    tiktoken = None

class EmbeddingService:
    def __init__(self):
        if settings.API_PROVIDER == "openai":
            self.provider = OpenAIService()
        elif settings.API_PROVIDER == "lmstudio":
            self.provider = LMStudioService()
        else:
            raise ValueError("Invalid API_PROVIDER in settings.")

    async def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Генерирует эмбеддинги для списка текстов.

        :param texts: Список текстовых строк.
        :return: Список numpy.ndarray с эмбеддингами.
        """
        embeddings = []
        for text in texts:
            embedding = await self.provider.generate_embedding(text)
            embeddings.append(np.array(embedding))
        return embeddings

    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Генерирует эмбеддинг для одного текста.

        :param text: Текстовая строка.
        :return: numpy.ndarray с эмбеддингом.
        """
        embedding = await self.provider.generate_embedding(text)
        return np.array(embedding)

    def num_tokens(self, text: str) -> int:
        """
        Подсчитывает количество токенов в тексте.

        :param text: Текстовая строка.
        :return: Количество токенов.
        """
        if tiktoken is None:
            raise ImportError("tiktoken library is not installed")
        encoding = tiktoken.encoding_for_model(settings.MODEL_NAME)
        tokens = encoding.encode(text)
        return len(tokens)

    def split_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Разбивает текст на чанки по количеству токенов.

        :param text: Текстовая строка.
        :param max_tokens: Максимальное количество токенов в чанке.
        :return: Список текстовых чанков.
        """
        if tiktoken is None:
            raise ImportError("tiktoken library is not installed")
        encoding = tiktoken.encoding_for_model(settings.MODEL_NAME)
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks

# Создаём экземпляр сервиса при импорте
embedding_service = EmbeddingService()
