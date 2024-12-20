# Файл: services/openai_service.py

import openai
from fastapi import HTTPException
import requests

from core.config import settings
from core.logger import logger

def get_models():
    """
    Получение списка доступных моделей OpenAI или LMStudio.
    """
    api_provider = settings.API_PROVIDER

    if api_provider == "openai":
        openai.api_key = settings.OPENAI_API_KEY
        try:
            response = openai.Model.list()
            models = [model["id"] for model in response["data"]]
            return models
        except openai.error.OpenAIError as e:
            logger.error(f"Ошибка OpenAI API: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")
    elif api_provider == "lmstudio":
        # Если LMStudio предоставляет список моделей, реализуйте здесь
        # Для примера вернём фиктивный список
        return ["lmstudio-default-model"]
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")

def generate_response(prompt: str):
    """
    Генерация ответа от модели OpenAI или LMStudio.
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
            logger.error(f"Ошибка OpenAI API: {e}")
            raise HTTPException(status_code=500, detail="OpenAI API error")
    elif api_provider == "lmstudio":
        # Реализация для LMStudio
        try:
            lmstudio_api_url = settings.LMSTUDIO_API_URL
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": prompt,
                "max_tokens": 150,   # Укажите нужное количество токенов
                "temperature": 0.7,  # Укажите нужное значение температуры
                # Добавьте другие параметры при необходимости
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
                logger.error(f"Ошибка LMStudio API: {response.text}")
                raise HTTPException(status_code=500, detail="LMStudio API error")
        except Exception as e:
            logger.error(f"Ошибка при обращении к LMStudio API: {e}")
            raise HTTPException(status_code=500, detail="LMStudio API error")
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")