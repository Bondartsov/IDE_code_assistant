# api.py

from uuid import uuid4
from math import ceil
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import openai
import logging
import numpy as np
from typing import List, Dict, Any
from io import BytesIO
import pypdf
import docx
import requests
import tiktoken

# Импортируем ConfigManager и FAISSIndexManager из соответствующих файлов
from config_storage import ConfigManager
from index_manager import FAISSIndexManager

# Инициализируем приложение FastAPI и менеджеры
app = FastAPI()
config_manager = ConfigManager()
index_manager = FAISSIndexManager(config_manager)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Словарь, содержащий максимальное количество токенов для каждой модели
MODEL_MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "chatgpt-4o-latest": 32768,  # Предположительное значение, уточните по документации OpenAI
    # Добавьте другие модели по необходимости
}

# Функция для подсчёта количества токенов в тексте
def num_tokens(text: str, model_name: str = None) -> int:
    """
    Возвращает количество токенов в тексте для заданной модели.

    :param text: Входной текст.
    :param model_name: Имя модели. Если не указано, используется модель из конфигурации.
    :return: Количество токенов.
    """
    if model_name is None:
        model_name = config_manager.get_model_name()
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# Функция для разделения текста на части с максимальным количеством токенов
def split_text(
    text: str, max_tokens: int, model_name: str = None
) -> List[str]:
    """
    Разбивает текст на куски, каждый из которых не превышает max_tokens токенов.

    :param text: Входной текст.
    :param max_tokens: Максимальное количество токенов в одном куске.
    :param model_name: Имя модели. Если не указано, используется модель из конфигурации.
    :return: Список текстовых кусков.
    """
    if model_name is None:
        model_name = config_manager.get_model_name()
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# Функция для обрезания текста до заданного количества токенов
def truncate_text_tokens(text: str, max_tokens: int, model_name: str = None) -> str:
    """
    Обрезает текст до max_tokens токенов.

    :param text: Входной текст.
    :param max_tokens: Максимальное количество токенов.
    :param model_name: Имя модели. Если не указано, используется модель из конфигурации.
    :return: Обрезанный текст.
    """
    if model_name is None:
        model_name = config_manager.get_model_name()
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    else:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text

# Модели данных для запросов
class OpenAIRequest(BaseModel):
    prompt: str  # Вопрос или запрос пользователя

class SearchRequest(BaseModel):
    query: str  # Поисковой запрос

class DocumentRequest(BaseModel):
    title: str  # Заголовок документа
    content: str = None  # Содержание документа (опционально)

# Функция для генерации эмбеддингов
def generate_embedding(text: str):
    """
    Генерирует эмбеддинг для заданного текста.

    :param text: Входной текст.
    :return: Эмбеддинг в виде списка чисел.
    """
    api_provider = config_manager.get_api_provider()
    embedding_model = "text-embedding-ada-002"  # Используем модель для эмбеддингов от OpenAI
    logging.info(f"Используем модель эмбеддингов '{embedding_model}'")  # Логирование

    if api_provider == "openai":
        # Используем OpenAI для генерации эмбеддингов
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key

        # Используем модель для эмбеддингов
        model_name = embedding_model

        # Разбиваем текст на куски, если это необходимо
        tokens = num_tokens(text, model_name=model_name)
        max_tokens = 8191  # Максимальное количество токенов для модели 'text-embedding-ada-002'
        if tokens > max_tokens:
            texts = split_text(text, max_tokens, model_name=model_name)
        else:
            texts = [text]

        embeddings = []
        for chunk in texts:
            try:
                response = openai.Embedding.create(
                    input=chunk, model=model_name
                )
                embedding = response["data"][0]["embedding"]
                embeddings.append(embedding)
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAIError: {e}")
                raise HTTPException(
                    status_code=500, detail=f"OpenAI API error: {str(e)}"
                )
        # Усредняем эмбеддинги, если было несколько кусков
        embedding = np.mean(embeddings, axis=0).tolist()
        return embedding

    elif api_provider == "lmstudio":
        # Используем LMStudio для генерации эмбеддингов
        lmstudio_url = config_manager.get_lmstudio_api_url()
        # Реализуйте логику генерации эмбеддингов с помощью LMStudio здесь
        # Примечание: Необходимо настроить модель и max_tokens в соответствии с требованиями LMStudio
        pass  # Замените этой строкой реальную реализацию

    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")

# Функция для генерации ответов
def generate_response(prompt: str):
    """
    Генерирует ответ на заданный запрос, используя выбранную модель.

    :param prompt: Запрос пользователя.
    :return: Сгенерированный ответ.
    """
    api_provider = config_manager.get_api_provider()
    model_name = config_manager.get_model_name()
    logging.info(f"Используем модель '{model_name}' для генерации ответа")  # Логирование
    if api_provider == "openai":
        # Используем OpenAI для генерации ответа
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content
            return response_text
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAIError: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    elif api_provider == "lmstudio":
        # Используем LMStudio для генерации ответа
        lmstudio_url = config_manager.get_lmstudio_api_url()
        try:
            response = requests.post(f"{lmstudio_url}/chat", json={"prompt": prompt})
            response.raise_for_status()
            response_text = response.json().get("response")
            if not response_text:
                raise ValueError("Response not found in LMStudio response")
            return response_text
        except Exception as e:
            logging.error(f"LMStudio error: {e}")
            raise HTTPException(status_code=500, detail=f"LMStudio API error: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")

# Эндпоинт для генерации нового внутреннего API-ключа
@app.post("/api/generate_key/")
def generate_api_key():
    """
    Генерирует новый API-ключ для доступа к API.
    """
    new_key = config_manager.generate_api_key()
    # Не логируем сгенерированный ключ по соображениям безопасности
    logging.info("Сгенерирован новый API-ключ")
    return {"api_key": new_key}

# Эндпоинт для отзыва существующего API-ключа
@app.post("/api/expire_key/{api_key}/")
def expire_api_key(api_key: str):
    """
    Отзывает (удаляет) указанный API-ключ.

    :param api_key: API-ключ, который необходимо отозвать.
    """
    if api_key in config_manager.valid_keys:
        del config_manager.valid_keys[api_key]
        logging.info(f"Отозван API-ключ: {api_key}")
        return {"detail": "API key expired successfully"}
    else:
        raise HTTPException(status_code=404, detail="API key not found")

# Эндпоинт для получения списка доступных моделей OpenAI
@app.get("/api/models/")
def get_models(api_key: str = Header(..., alias="api-key")):
    """
    Возвращает список доступных моделей OpenAI.

    :param api_key: Действительный API-ключ.
    """
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key
        # Получаем список моделей из API OpenAI
        response = openai.Model.list()
        models = [model["id"] for model in response["data"]]
        return {"models": models}
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")

# Эндпоинт для выполнения запроса к модели
@app.post("/api/openai/")
def run_openai_prompt(
    request: OpenAIRequest, api_key: str = Header(..., alias="api-key")
):
    """
    Выполняет запрос к модели с учётом контекста из базы знаний.

    :param request: Запрос пользователя с полем 'prompt'.
    :param api_key: Действительный API-ключ.
    """
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    prompt = request.prompt

    # Получаем имя модели и максимальное количество токенов
    model_name = config_manager.get_model_name()
    max_model_tokens = MODEL_MAX_TOKENS.get(model_name, 4096)  # Значение по умолчанию 4096 токенов
    reserved_tokens = 1000  # Резервируем токены для ответа
    max_prompt_tokens = max_model_tokens - reserved_tokens

    # Генерируем эмбеддинг для запроса
    query_embedding = generate_embedding(prompt)

    # Поиск в индексе
    results = index_manager.search(np.array(query_embedding), top_k=5)

    # Формируем контекст
    context = ""
    if results:
        # Сортируем результаты по document_id или другим критериям
        # Здесь используется сортировка по 'doc_id' для консистентности
        results.sort(key=lambda x: x["doc_id"])
        context = "\n\n".join([doc["content"] for doc in results])

        # Вычисляем доступное количество токенов для контекста
        prompt_tokens = num_tokens(prompt, model_name=model_name)
        overhead_tokens = num_tokens("Context:\n\nQuestion:\n", model_name=model_name)
        available_context_tokens = max_prompt_tokens - prompt_tokens - overhead_tokens

        if available_context_tokens > 0:
            # Обрезаем контекст, если он превышает доступное количество токенов
            context = truncate_text_tokens(
                context, available_context_tokens, model_name=model_name
            )
            final_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
        else:
            # Недостаточно места для контекста
            final_prompt = prompt
    else:
        final_prompt = prompt

    try:
        # Генерируем ответ с использованием функции generate_response
        response_text = generate_response(final_prompt)
        return {"response": response_text}
    except Exception as e:
        logging.error(f"Error during completion: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during completion: {str(e)}"
        )

# Функция для извлечения текста из PDF-файла
async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Извлекает текст из загруженного PDF-файла.

    :param file: Загруженный файл.
    :return: Извлечённый текст.
    """
    contents = await file.read()
    reader = pypdf.PdfReader(BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Функция для извлечения текста из DOCX-файла
async def extract_text_from_docx(file: UploadFile) -> str:
    """
    Извлекает текст из загруженного DOCX-файла.

    :param file: Загруженный файл.
    :return: Извлечённый текст.
    """
    contents = await file.read()
    document = docx.Document(BytesIO(contents))
    text = "\n".join([para.text for para in document.paragraphs])
    return text

# Эндпоинт для добавления документа в базу знаний
@app.post("/api/knowledge_base/")
async def add_document_endpoint(
    api_key: str = Header(..., alias="api-key"),
    title: str = Form(...),
    content: str = Form(None),
    file: UploadFile = File(None),
):
    """
    Добавляет документ в базу знаний.

    :param api_key: Действительный API-ключ.
    :param title: Заголовок документа.
    :param content: Содержание документа (опционально).
    :param file: Загруженный файл (опционально).
    """
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if content is None and file is None:
        raise HTTPException(
            status_code=400, detail="Either content or file must be provided"
        )

    if file is not None:
        # Максимальный размер файла для загрузки (например, 50 МБ)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 МБ
        # Проверяем размер файла
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, detail="File size exceeds the maximum limit."
            )
        # Сбрасываем указатель файла после чтения
        file.file.seek(0)
        # Обрабатываем файл в зависимости от его типа
        if file.content_type == "application/pdf":
            content = await extract_text_from_pdf(file)
        elif (
            file.content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            content = await extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Проверяем количество токенов в тексте
    tokens = num_tokens(content)
    max_tokens = MODEL_MAX_TOKENS.get(config_manager.get_model_name(), 4096) - 1000  # Резервируем токены
    if tokens > max_tokens:
        # Если текст превышает максимальный размер, разбиваем его на части
        content_chunks = split_text(content, max_tokens)
    else:
        content_chunks = [content]

    # Генерируем уникальный идентификатор для всего документа
    document_id = str(uuid4())

    # Добавляем каждую часть в базу данных и индекс
    doc_ids = []
    for idx, chunk in enumerate(content_chunks):
        chunk_title = f"{title} (Part {idx+1})" if len(content_chunks) > 1 else title
        # Генерируем эмбеддинг для куска
        embedding = generate_embedding(chunk)
        embedding_bytes = np.array(embedding, dtype="float32").tobytes()
        # Сохраняем кусок в базе данных
        doc_id = config_manager.add_document(
            chunk_title, chunk, embedding_bytes, document_id=document_id
        )
        # Добавляем в индекс
        index_manager.add_document(
            doc_id,
            np.array(embedding),
            {"title": chunk_title, "content": chunk}
        )
        doc_ids.append(doc_id)

    return {"detail": "Document added successfully", "doc_ids": doc_ids}

# Эндпоинт для поиска по базе знаний
@app.post("/api/search/")
def search_endpoint(
    request: SearchRequest, api_key: str = Header(..., alias="api-key")
):
    """
    Выполняет поиск по базе знаний.

    :param request: Поисковой запрос с полем 'query'.
    :param api_key: Действительный API-ключ.
    """
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    query = request.query

    # Генерируем эмбеддинг для запроса
    query_embedding = generate_embedding(query)

    # Поиск в индексе
    results = index_manager.search(np.array(query_embedding), top_k=5)

    # Группируем результаты по document_id
    grouped_results = {}
    for result in results:
        doc_id = result.get("document_id")
        if doc_id not in grouped_results:
            grouped_results[doc_id] = {"title": result["title"], "content": ""}
        grouped_results[doc_id]["content"] += result["content"] + "\n\n"

    # Преобразуем в список
    final_results = list(grouped_results.values())

    # Формируем ответ
    return {"results": final_results}