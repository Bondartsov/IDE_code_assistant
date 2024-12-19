# Файл: api.py

import os
from fastapi import FastAPI, HTTPException, Header
import openai
import logging
from config_storage import ConfigManager
from pydantic import BaseModel

# Загрузка переменных окружения из .env файла
from dotenv import load_dotenv
load_dotenv()

# Инициализируем ConfigManager и устанавливаем API-ключ OpenAI
config_manager = ConfigManager()
openai_api_key = config_manager.get_openai_api_key()
openai.api_key = openai_api_key  # Устанавливаем API-ключ OpenAI

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)

# Модель запроса для OpenAI
class OpenAIRequest(BaseModel):
    prompt: str

# Эндпоинт для генерации нового внутреннего API-ключа
@app.post("/api/generate_key/")
def generate_api_key():
    new_key = config_manager.generate_api_key()
    # Не выводим сгенерированный ключ в логи по соображениям безопасности
    logging.info("Generated new API key")
    return {"api_key": new_key}

# Эндпоинт для получения списка доступных моделей OpenAI
@app.get("/api/models/")
def get_models(api_key: str = Header(..., alias="api-key")):
    # Проверка валидности внутреннего API-ключа
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        # Получаем список моделей из OpenAI API
        response = openai.Model.list()
        models = [model["id"] for model in response["data"]]
        return {"models": models}
    except openai.error.OpenAIError as e:
        # Логируем ошибку без раскрытия подробностей
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")
    except Exception as e:
        # Логируем неожиданные ошибки
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

# Эндпоинт для выполнения запроса к OpenAI API
@app.post("/api/openai/")
def run_openai_prompt(request: OpenAIRequest, api_key: str = Header(..., alias="api-key")):
    # Валидируем внутренний API-ключ
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Продолжаем без изменения openai.api_key
    prompt = request.prompt
    try:
        # Отправляем запрос к OpenAI API
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Убедитесь, что выбранная модель доступна
            messages=[{"role": "user", "content": prompt}]
        )
        # Возвращаем ответ от модели
        return {"response": result.choices[0].message.content}
    except openai.error.OpenAIError as e:
        # Логируем ошибки OpenAI
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")
    except Exception as e:
        # Логируем неожиданные ошибки
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")