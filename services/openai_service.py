# services/openai_service.py

"""
Модуль для взаимодействия с OpenAI и LMStudio.
"""

import openai
from fastapi import HTTPException
import requests

from core.config import settings
from core.logger import logger

def get_models() -> list[str]:
    """
    Возвращает список доступных моделей.

    Returns:
        list[str]: Список идентификаторов моделей.

    Raises:
        HTTPException: При ошибке доступа к API.
    """
    api_provider = settings.API_PROVIDER

    if api_provider == "openai":
        openai.api_key = settings.OPENAI_API_KEY
        try:
            response = openai.Model.list()
            models = [model["id"] for model in response["data"]]
            return models
        except openai.error.OpenAIError as e:
            logger.error(f"Error accessing OpenAI API: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")
    elif api_provider == "lmstudio":
        # Предполагаем, что LMStudio имеет аналогичный API
        try:
            lmstudio_api_url = f"{settings.LMSTUDIO_API_URL}/models"
            response = requests.get(lmstudio_api_url)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return models
            else:
                logger.error(f"Error from LMStudio API: {response.text}")
                raise HTTPException(status_code=500, detail="LMStudio API error")
        except Exception as e:
            logger.error(f"Error accessing LMStudio API: {e}")
            raise HTTPException(status_code=500, detail="LMStudio API error")
    else:
        logger.error("Unsupported API provider")
        raise HTTPException(status_code=500, detail="Unsupported API provider")

def generate_response(prompt: str) -> str:
    """
    Генерирует ответ от модели на заданный промпт.

    Args:
        prompt (str): Промпт для модели.

    Returns:
        str: Ответ модели.

    Raises:
        HTTPException: При ошибке доступа к API.
    """
    api_provider = settings.API_PROVIDER
    model_name = settings.MODEL_NAME

    if api_provider == "openai":
        openai.api_key = settings.OPENAI_API_KEY
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content
            return response_text
        except openai.error.OpenAIError as e:
            logger.error(f"Error accessing OpenAI API: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")
    elif api_provider == "lmstudio":
        try:
            lmstudio_api_url = settings.LMSTUDIO_API_URL
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.7,
            }
            response = requests.post(
                lmstudio_api_url,
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                response_text = response.json()["choices"][0]["text"]
                return response_text.strip()
            else:
                logger.error(f"Error from LMStudio API: {response.text}")
                raise HTTPException(status_code=500, detail="LMStudio API error")
        except Exception as e:
            logger.error(f"Error accessing LMStudio API: {e}")
            raise HTTPException(status_code=500, detail="LMStudio API error")
    else:
        logger.error("Unsupported API provider")
        raise HTTPException(status_code=500, detail="Unsupported API provider")