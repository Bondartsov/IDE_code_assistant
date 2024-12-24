import openai
from fastapi import HTTPException
from core.config import settings
from core.logger import logger
from typing import List


class OpenAIService:
    """
    Класс для работы с OpenAI API.
    """

    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для текста с использованием OpenAI API.

        :param text: Текст для генерации эмбеддинга.
        :return: Эмбеддинг в виде списка чисел.
        """
        try:
            embedding_model = settings.EMBEDDING_MODEL_NAME
            response = await openai.Embedding.acreate(
                input=text,
                model=embedding_model
            )
            embedding = response["data"][0]["embedding"]
            return embedding
        except openai.error.OpenAIError as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")

    async def generate_response(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Генерирует ответ с использованием OpenAI API.

        :param prompt: Текстовый запрос, включающий контекст и вопрос пользователя.
        :param model: Название модели для генерации ответа.
        :return: Сгенерированный текст.
        """
        try:
            completion = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content
            return response_text
        except openai.error.OpenAIError as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")

    async def get_models(self) -> List[str]:
        """
        Возвращает список доступных моделей OpenAI.

        :return: Список названий моделей.
        """
        try:
            response = await openai.Model.alist()
            models = [model["id"] for model in response["data"]]
            return models
        except openai.error.OpenAIError as e:
            logger.error(f"Error fetching models: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")