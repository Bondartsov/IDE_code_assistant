from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from typing import List, Optional, Union
from sqlalchemy.orm import Session
from services.file_service import extract_text
from services.embedding_service import generate_embedding, num_tokens, split_text
from services.indexing_service import index_manager
from services.openai_service import generate_response, get_models

from core.security import validate_api_key, api_key_manager
from core.logger import logger
from core.database import get_db, Document as DocumentModel
from api.models import OpenAIRequestModel

router = APIRouter()

@router.post("/generate_key/")
async def generate_key_endpoint():
    """
    Generates a new API key.
    """
    new_key = api_key_manager.create_api_key()
    logger.info("New API key generated")
    return {"api_key": new_key}

@router.post("/expire_key/{api_key}/")
async def expire_key_endpoint(api_key: str):
    """
    Expires the specified API key.
    Args:
        api_key (str): The API key to expire.
    """
    api_key_manager.invalidate_api_key(api_key)
    logger.info(f"API key {api_key} expired")
    return {"detail": "API key expired"}

@router.get("/models/")
async def get_models_endpoint(api_key: str = Depends(validate_api_key)):
    """
    Returns a list of available models.
    """
    models = get_models()
    return {"models": models}

@router.post("/openai/")
async def openai_endpoint(
    request: OpenAIRequestModel,
    api_key: str = Depends(validate_api_key)
):
    """
    Generates a response from the model for the given prompt.
    Args:
        request (OpenAIRequestModel): The request containing the prompt.
    """
    response = generate_response(request.prompt)
    return {"response": response}

@router.post("/knowledge_base/")
async def add_to_knowledge_base(
    files: Optional[List[UploadFile]] = File(None),
    texts: Union[str, List[str], None] = Form(None),
    api_key: str = Depends(validate_api_key)
):
    """
    Эндпоинт для добавления данных в базу знаний.
    """
    # Нормализация texts в список
    if texts is None:
        texts = []
    elif isinstance(texts, str):
        texts = [texts]

    if not files and not texts:
        raise HTTPException(status_code=400, detail="No files or texts provided")
    
    combined_text = await extract_text(files, texts)

    # Дополнительная обработка combined_text...
    # Разбиение текста на чанки, генерация эмбеддингов, добавление в индекс и т.д.
    logger.info("Data added to the knowledge base")
    return {"message": "Data added to the knowledge base"}

@router.post("/search/")
async def search_documents(
    query: OpenAIRequestModel,
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    Searches documents in the knowledge base based on the given query.
    Args:
        query (OpenAIRequestModel): The search query.
    """
    query_embedding = generate_embedding(query.prompt)
    search_results = index_manager.search(query_embedding)

    document_ids = [result['doc_id'] for result in search_results]
    documents = db.query(DocumentModel).filter(DocumentModel.id.in_(document_ids)).all()

    results = []
    for doc in documents:
        results.append({
            "id": doc.id,
            "content": doc.content
        })

    return {"results": results}