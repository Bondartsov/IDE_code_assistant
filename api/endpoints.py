# api/endpoints.py

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKey
from sqlalchemy.orm import Session
from api.models import DocumentBase, DocumentCreate, OpenAIRequestModel
from services.indexing_service import index_manager
from services.embedding_service import generate_embedding
from core.security import get_api_key
from core.logger import logger
from core.database import get_db, Document as DocumentModel
from services.openai_service import generate_response, get_models

router = APIRouter()

@router.post("/generate_key/")
async def generate_key_endpoint():
    """
    Генерация нового API-ключа.
    """
    from core.security import api_key_manager
    new_key = api_key_manager.create_api_key()
    logger.info("Сгенерирован новый API-ключ")
    return {"api_key": new_key}

@router.post("/expire_key/{api_key}/")
async def expire_key_endpoint(api_key: str):
    """
    Отзыв API-ключа.
    """
    from core.security import api_key_manager
    api_key_manager.invalidate_api_key(api_key)
    logger.info(f"API-ключ {api_key} отозван")
    return {"detail": "API key expired"}

@router.get("/models/")
async def get_models_endpoint(api_key: APIKey = Security(get_api_key)):
    """
    Получение списка доступных моделей.
    """
    models = get_models()
    return {"models": models}

@router.post("/openai/")
async def openai_endpoint(request: OpenAIRequestModel, api_key: APIKey = Security(get_api_key)):
    """
    Выполнение запроса к OpenAI.
    """
    response = generate_response(request.prompt)
    return {"response": response}

@router.post("/knowledge_base/")
async def add_document(
    document: DocumentBase,
    db: Session = Depends(get_db),
    api_key: APIKey = Security(get_api_key)
):
    """
    Добавление документа в базу знаний.
    """
    # Сохранение документа в базе данных
    db_document = DocumentModel(title=document.title, content=document.content)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Генерация эмбеддинга
    embedding = generate_embedding(document.content)
    # Добавление эмбеддинга в индекс
    index_manager.add_document(db_document.id, embedding)

    logger.info(f"Документ {db_document.id} добавлен в базу знаний")

    return {"detail": "Document added successfully", "document_id": db_document.id}

@router.post("/search/")
async def search_documents(
    query: OpenAIRequestModel,
    db: Session = Depends(get_db),
    api_key: APIKey = Security(get_api_key)
):
    """
    Поиск документов в базе знаний.
    """
    # Генерация эмбеддинга для запроса
    query_embedding = generate_embedding(query.prompt)
    # Поиск похожих документов в индексе
    search_results = index_manager.search(query_embedding)

    # Получение документов из базы данных
    document_ids = [result['doc_id'] for result in search_results]
    documents = db.query(DocumentModel).filter(DocumentModel.id.in_(document_ids)).all()

    # Формирование результатов
    results = []
    for doc in documents:
        results.append({"id": doc.id, "title": doc.title, "content": doc.content})

    return {"results": results}