# api/endpoints.py

from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile, File
from fastapi.security.api_key import APIKey
from sqlalchemy.orm import Session

from api.models import DocumentBase, DocumentCreate, OpenAIRequestModel
from services.indexing_service import index_manager
from services.embedding_service import generate_embedding
from services.openai_service import generate_response, get_models
from services.file_service import extract_text
from core.security import get_api_key, api_key_manager
from core.logger import logger
from core.database import get_db, Document as DocumentModel

router = APIRouter()

@router.post("/generate_key/")
async def generate_key_endpoint():
    """
    Генерирует новый API-ключ.
    """
    new_key = api_key_manager.create_api_key()
    logger.info("New API key generated")
    return {"api_key": new_key}

@router.post("/expire_key/{api_key}/")
async def expire_key_endpoint(api_key: str):
    """
    Аннулирует указанный API-ключ.

    Args:
        api_key (str): API-ключ для аннулирования.
    """
    api_key_manager.invalidate_api_key(api_key)
    logger.info(f"API key {api_key} expired")
    return {"detail": "API key expired"}

@router.get("/models/")
async def get_models_endpoint(api_key: APIKey = Security(get_api_key)):
    """
    Возвращает список доступных моделей.
    """
    models = get_models()
    return {"models": models}

@router.post("/openai/")
async def openai_endpoint(
    request: OpenAIRequestModel,
    api_key: APIKey = Security(get_api_key)
):
    """
    Генерирует ответ от модели на заданный промпт.

    Args:
        request (OpenAIRequestModel): Запрос с промптом.
    """
    response = generate_response(request.prompt)
    return {"response": response}

@router.post("/knowledge_base/")
async def add_document(
    title: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: APIKey = Security(get_api_key)
):
    """
    Добавляет документ в базу знаний.

    Args:
        title (str): Заголовок документа.
        file (UploadFile): Загруженный файл.
    """
    try:
        content = await extract_text(file)
    except ValueError as e:
        logger.error(f"File extraction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    db_document = DocumentModel(title=title, content=content)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    embedding = generate_embedding(content)
    index_manager.add_document(db_document.id, embedding)

    logger.info(f"Document {db_document.id} added to knowledge base")

    return {"detail": "Document added successfully", "document_id": db_document.id}

@router.post("/search/")
async def search_documents(
    query: OpenAIRequestModel,
    db: Session = Depends(get_db),
    api_key: APIKey = Security(get_api_key)
):
    """
    Ищет документы в базе знаний по заданному запросу.

    Args:
        query (OpenAIRequestModel): Запрос для поиска.
    """
    query_embedding = generate_embedding(query.prompt)
    search_results = index_manager.search(query_embedding)

    document_ids = [result['doc_id'] for result in search_results]
    documents = db.query(DocumentModel).filter(DocumentModel.id.in_(document_ids)).all()

    results = []
    for doc in documents:
        results.append({
            "id": doc.id,
            "title": doc.title,
            "content": doc.content
        })

    return {"results": results}