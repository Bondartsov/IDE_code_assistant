# Файл: api/endpoints.py

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File, Form
from typing import List

from core.security import get_current_api_key
from core.logger import logger
from services.openai_service import generate_response, get_models
from services.embedding_service import generate_embedding
from services.indexing_service import index_manager
from services.file_service import extract_text_from_pdf, extract_text_from_docx
from core.config import settings
from core.database import get_db, SessionLocal, Document
from api.models import DocumentBase
from uuid import uuid4
import numpy as np

router = APIRouter()

@router.post("/api/generate_key/")
def generate_api_key():
    """
    Генерация нового внутреннего API-ключа.
    """
    from core.security import api_key_manager
    new_key = api_key_manager.create_api_key()
    logger.info("Сгенерирован новый API-ключ")
    return {"api_key": new_key}

@router.post("/api/expire_key/{api_key}/")
def expire_api_key(api_key: str):
    """
    Отзыв (удаление) существующего API-ключа.
    """
    from core.security import api_key_manager
    if api_key_manager.invalidate_api_key(api_key):
        logger.info(f"API-ключ {api_key} отозван")
        return {"detail": "API key expired successfully"}
    else:
        raise HTTPException(status_code=404, detail="API key not found")

@router.get("/api/models/")
def get_available_models(api_key: str = Depends(get_current_api_key)):
    """
    Получение списка доступных моделей OpenAI.
    """
    models = get_models()
    return {"models": models}

@router.post("/api/openai/")
def run_openai_prompt(
    prompt: str = Form(..., description="Текст запроса для модели"),
    api_key: str = Depends(get_current_api_key)
):
    """
    Выполнение запроса к модели OpenAI или LMStudio.
    """
    response_text = generate_response(prompt)
    return {"response": response_text}

@router.post("/api/knowledge_base/")
async def add_document_endpoint(
    title: str = Form(..., description="Заголовок документа"),
    content: str = Form(None, description="Содержание документа"),
    file: UploadFile = File(None, description="Файл для загрузки"),
    api_key: str = Depends(get_current_api_key)
):
    """
    Добавление документа в базу знаний.
    """
    from services.embedding_service import num_tokens, split_text
    if content is None and file is None:
        raise HTTPException(
            status_code=400, detail="Either content or file must be provided"
        )

    if file is not None:
        # Проверка типа файла и извлечение текста
        if file.content_type == "application/pdf":
            content = await extract_text_from_pdf(file)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = await extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Проверка количества токенов
    tokens = num_tokens(content)
    max_tokens = 4000  # Установите соответствующее значение
    if tokens > max_tokens:
        content_chunks = split_text(content, max_tokens)
    else:
        content_chunks = [content]

    document_id = str(uuid4())
    db = SessionLocal()
    for idx, chunk in enumerate(content_chunks):
        chunk_title = f"{title} (Part {idx+1})" if len(content_chunks) > 1 else title
        embedding = generate_embedding(chunk)
        document = Document(
            document_id=document_id,
            title=chunk_title,
            content=chunk,
            embedding=np.array(embedding).astype('float32').tobytes()
        )
        db.add(document)
        db.commit()
        doc_id = document.id
        index_manager.add_document(doc_id, embedding)
    db.close()
    return {"detail": "Document added successfully", "document_id": document_id}

@router.post("/api/search/")
def search_endpoint(
    query: str = Form(..., description="Поисковый запрос"),
    api_key: str = Depends(get_current_api_key)
):
    """
    Поиск по базе знаний.
    """
    query_embedding = generate_embedding(query)
    results = index_manager.search(query_embedding)
    return {"results": results}