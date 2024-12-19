# Файл: api.py

import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Header
import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from config_storage import ConfigManager
from index_manager import FAISSIndexManager
import tiktoken  # Добавлено для работы с токенами

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализируем менеджеры
config_manager = ConfigManager()
index_manager = FAISSIndexManager()

# Устанавливаем API-ключ OpenAI
openai_api_key = config_manager.get_openai_api_key()
openai.api_key = openai_api_key

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)

# Модели Pydantic для запросов
class OpenAIRequest(BaseModel):
    prompt: str

class DocumentRequest(BaseModel):
    title: str
    content: str

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
    prompt = request.prompt
    # Получаем эмбеддинг запроса
    openai_api_key = config_manager.get_openai_api_key()
    openai.api_key = openai_api_key

    try:
        embedding_response = openai.Embedding.create(
            input=prompt,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Ищем релевантные документы
    try:
        relevant_docs = index_manager.search(np.array(query_embedding), top_k=5)
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search error occurred")

    # Формируем контекст из найденных документов
    context = ""
    if relevant_docs:
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    else:
        # Если нет релевантных документов, используем исходный prompt
        final_prompt = prompt

    # Проверяем длину prompt и сокращаем при необходимости
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(final_prompt))
    MAX_TOKENS = 4096
    if num_tokens > MAX_TOKENS:
        # Суммаризируем контекст, если он есть
        if context:
            summary_prompt = f"Please summarize the following text:\n\n{context}"
            try:
                summary_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=1024
                )
                summarized_context = summary_response.choices[0].message.content
                final_prompt = f"Context:\n{summarized_context}\n\nQuestion:\n{prompt}"
            except openai.error.OpenAIError as e:
                raise HTTPException(status_code=500, detail=f"OpenAI API error during summarization: {str(e)}")
        else:
            # Если контекст отсутствует, используем исходный prompt
            final_prompt = prompt

    # Отправляем окончательный prompt в OpenAI API
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": final_prompt}]
        )
        response_text = completion.choices[0].message.content
        return {"response": response_text}
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

# Эндпоинт для добавления документа в базу знаний

@app.post("/api/knowledge_base/")
def add_document_endpoint(request: DocumentRequest, api_key: str = Header(..., alias="api-key")):
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Получаем эмбеддинг контента с помощью OpenAI
    openai_api_key = config_manager.get_openai_api_key()
    openai.api_key = openai_api_key

    try:
        response = openai.Embedding.create(
            input=request.content,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Сохраняем документ и эмбеддинг
    embedding_bytes = np.array(embedding, dtype='float32').tobytes()
    # Исправление: заменяем save_document на add_document
    doc_id = config_manager.add_document(request.title, request.content, embedding_bytes)
    index_manager.add_document(doc_id, np.array(embedding), {'title': request.title})

    return {"detail": "Document added successfully", "doc_id": doc_id}
