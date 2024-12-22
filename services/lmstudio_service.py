# services/lmstudio_service.py

import httpx
from core.config import settings

class LMStudioService:
    """
    Класс для взаимодействия с локальной моделью через LMStudio API.
    """

    def __init__(self):
        self.api_url = settings.LMSTUDIO_API_URL
        self.embedding_api_url = settings.LMSTUDIO_EMBEDDING_API_URL

    async def generate_embedding(self, text: str):
        """
        Генерирует эмбеддинг текста с помощью LMStudio.

        :param text: Текст для генерации эмбеддинга.
        :return: Эмбеддинг в виде списка чисел.
        """
        if not self.embedding_api_url:
            raise NotImplementedError("LMSTUDIO_EMBEDDING_API_URL не задан в настройках.")
        payload = {
            "text": text,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.embedding_api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["embedding"]

    async def generate_completion(self, prompt: str):
        """
        Генерирует ответ модели на заданный промпт.

        :param prompt: Входной текст или вопрос.
        :return: Ответ модели.
        """
        payload = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": None,
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
