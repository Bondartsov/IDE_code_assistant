# services/openai_service.py

import openai
from fastapi import HTTPException
from core.config import settings
from core.logger import logger
from typing import List  # Import List for type hinting

class OpenAIService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY

    async def generate_embedding(self, text: str):
        """
        Генерирует эмбеддинг для заданного текста.
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

    async def generate_response(self, prompt: str) -> str:
        """
        Генерирует ответ модели для заданного промпта.
        """
        model_name = settings.MODEL_NAME
        try:
            completion = await openai.ChatCompletion.acreate(
                model=model_name,
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
        """
        try:
            response = await openai.Model.alist()
            models = [model["id"] for model in response["data"]]
            return models
        except openai.error.OpenAIError as e:
            logger.error(f"Error fetching models: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")