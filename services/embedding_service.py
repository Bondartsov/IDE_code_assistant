# services/embedding_service.py

from typing import List
import numpy as np
from core.config import settings
from services.openai_service import OpenAIService
from services.lmstudio_service import LMStudioService
from core.logger import logger

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

embedding_service = EmbeddingService()
