# api/endpoints.py

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from core.security import validate_api_key
from services.file_service import process_files, process_text_data
from services.indexing_service import indexing_service
from services.embedding_service import embedding_service
from api.models import APIKeyName, OpenAIRequestModel
from pydantic import BaseModel

router = APIRouter()

class SearchRequest(BaseModel):
    """
    Модель запроса для поиска.
    """
    query: str

class SearchResult(BaseModel):
    """
    Модель результата поиска.
    """
    id: int
    title: str
    content: str

@router.post("/generate_key/")
def generate_api_key(api_key_name: APIKeyName):
    """
    Генерирует новый API-ключ для использования в запросах.
    """
    from core.security import api_key_manager
    api_key = api_key_manager.create_api_key()
    return {"api_key": api_key}

@router.post("/knowledge_base/")
async def upload_to_knowledge_base(
    files: List[UploadFile] = File(None),
    text_data: List[str] = None,
    api_key: str = Depends(validate_api_key),
):
    """
    Загружает файлы и текстовые данные в базу знаний.
    """
    # Обрабатываем файлы и текстовые данные
    texts = []
    if files:
        texts.extend(await process_files(files))
    if text_data:
        texts.extend(process_text_data(text_data))

    if not texts:
        raise HTTPException(status_code=400, detail="No text data or files provided.")

    # Генерируем эмбеддинги
    embeddings = await embedding_service.generate_embeddings(texts)

    # Добавляем документы и эмбеддинги в индекс
    indexing_service.add_documents(texts, embeddings)

    return {"message": "Данные успешно добавлены в базу знаний."}

@router.post("/search/", response_model=List[SearchResult])
async def search_knowledge_base(
    request: SearchRequest,
    api_key: str = Depends(validate_api_key),
):
    """
    Ищет по базе знаний и возвращает наиболее релевантные результаты.
    """
    query_embedding = await embedding_service.generate_embedding(request.query)
    top_k = 5  # Количество результатов для возврата

    results = indexing_service.search(query_embedding, top_k=top_k)
    if not results:
        return []

    # Извлекаем документы по идентификаторам
    documents = indexing_service.get_documents_by_ids(results)

    return documents

@router.post("/openai/")
async def run_openai_prompt(
    request: OpenAIRequestModel,
    api_key: str = Depends(validate_api_key),
):
    """
    Отправляет промпт в модель OpenAI и возвращает ответ.
    """
    from services.openai_service import OpenAIService
    openai_service = OpenAIService()
    response_text = await openai_service.generate_response(request.prompt)
    return {"response": response_text}

@router.get("/models/", response_model=List[str])
async def get_models(api_key: str = Depends(validate_api_key)):
    """
    Возвращает список доступных моделей OpenAI.
    """
    from services.openai_service import OpenAIService
    openai_service = OpenAIService()
    models = await openai_service.get_models()
    return models