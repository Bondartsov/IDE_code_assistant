# File: api.py

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import openai
import logging
import numpy as np
from typing import List, Dict, Any
from io import BytesIO
import pypdf
import docx

# Import managers
from config_storage import ConfigManager
from index_manager import FAISSIndexManager

# Initialize managers
config_manager = ConfigManager()
index_manager = FAISSIndexManager()

# Create FastAPI app instance
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Pydantic models for requests
class OpenAIRequest(BaseModel):
    prompt: str

class SearchRequest(BaseModel):
    query: str

# Endpoint to generate a new internal API key
@app.post("/api/generate_key/")
def generate_api_key():
    new_key = config_manager.generate_api_key()
    # Do not log the generated key for security reasons
    logging.info("Generated new API key")
    return {"api_key": new_key}

# Endpoint to expire an existing API key
@app.post("/api/expire_key/{api_key}/")
def expire_api_key(api_key: str):
    if api_key in config_manager.valid_keys:
        del config_manager.valid_keys[api_key]
        logging.info(f"Expired API key: {api_key}")
        return {"detail": "API key expired successfully"}
    else:
        raise HTTPException(status_code=404, detail="API key not found")

# Endpoint to get the list of available OpenAI models
@app.get("/api/models/")
def get_models(api_key: str = Header(..., alias="api-key")):
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        openai.api_key = config_manager.get_openai_api_key()
        # Get the list of models from OpenAI API
        response = openai.Model.list()
        models = [model["id"] for model in response["data"]]
        return {"models": models}
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")

# Endpoint to make a request to the OpenAI API
@app.post("/api/openai/")
def run_openai_prompt(
    request: OpenAIRequest, api_key: str = Header(..., alias="api-key")
):
    # Validate internal API key
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    prompt = request.prompt

    # Set OpenAI API key
    openai.api_key = config_manager.get_openai_api_key()

    # Get embedding of the prompt
    try:
        embedding_response = openai.Embedding.create(
            input=prompt, model="text-embedding-ada-002"
        )
        query_embedding = embedding_response["data"][0]["embedding"]
    except openai.error.OpenAIError as e:
        raise HTTPException(
            status_code=500, detail=f"OpenAI API error: {str(e)}"
        )

    # Search for relevant documents
    try:
        relevant_docs = index_manager.search(
            np.array(query_embedding), top_k=5
        )
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500, detail="Search error occurred"
        )

    # Build context from found documents
    context = ""
    if relevant_docs:
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    else:
        # If no relevant documents, use the original prompt
        final_prompt = prompt

    # Check prompt length and truncate if necessary
    MAX_TOKENS = 4096
    total_tokens_estimated = len(final_prompt.split())  # Rough estimation
    if total_tokens_estimated > MAX_TOKENS:
        # Summarize context if it exists
        if context:
            summary_prompt = f"Please summarize the following text:\n\n{context}"
            try:
                summary_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=1024,
                )
                summarized_context = summary_response.choices[0].message.content
                final_prompt = (
                    f"Context:\n{summarized_context}\n\nQuestion:\n{prompt}"
                )
            except openai.error.OpenAIError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API error during summarization: {str(e)}",
                )
        else:
            # If no context, use the original prompt
            pass  # Do nothing

    try:
        # Send the final prompt to OpenAI API
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": final_prompt}],
        )
        response_text = completion.choices[0].message.content
        return {"response": response_text}
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(
            status_code=500, detail="OpenAI API error"
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, detail="Unexpected error occurred"
        )

# Function to extract text from PDF using pypdf
async def extract_text_from_pdf(file: UploadFile) -> str:
    reader = pypdf.PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
async def extract_text_from_docx(file: UploadFile) -> str:
    contents = await file.read()
    document = docx.Document(BytesIO(contents))
    text = "\n".join([para.text for para in document.paragraphs])
    return text

# Endpoint to add a document to the knowledge base
@app.post("/api/knowledge_base/")
async def add_document_endpoint(
    api_key: str = Header(..., alias="api-key"),
    title: str = Form(...),
    content: str = Form(None),
    file: UploadFile = File(None)
):
    # Validate internal API key
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if content is None and file is None:
        raise HTTPException(status_code=400, detail="Either content or file must be provided")

    if file is not None:
        # Process uploaded file
        if file.content_type == "application/pdf":
            content = await extract_text_from_pdf(file)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = await extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Check that content is not empty after processing the file
    if not content:
        raise HTTPException(status_code=400, detail="Content is empty after processing the file")

    # Set OpenAI API key
    openai_api_key = config_manager.get_openai_api_key()
    openai.api_key = openai_api_key

    # Get embedding of the content using OpenAI
    try:
        response = openai.Embedding.create(
            input=content,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Save document and embedding
    embedding_bytes = np.array(embedding, dtype='float32').tobytes()
    doc_id = config_manager.add_document(title, content, embedding_bytes)
    index_manager.add_document(doc_id, np.array(embedding), {'title': title})

    return {"detail": "Document added successfully", "doc_id": doc_id}

# Endpoint to search the knowledge base
@app.post("/api/search/")
def search_endpoint(
    request: SearchRequest, api_key: str = Header(..., alias="api-key")
):
    # Validate internal API key
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    query = request.query

    # Set OpenAI API key
    openai.api_key = config_manager.get_openai_api_key()

    # Get embedding of the query
    try:
        response = openai.Embedding.create(
            input=query, model="text-embedding-ada-002"
        )
        query_embedding = response["data"][0]["embedding"]
    except openai.error.OpenAIError as e:
        raise HTTPException(
            status_code=500, detail=f"OpenAI API error: {str(e)}"
        )

    # Perform search in the index
    try:
        results = index_manager.search(
            np.array(query_embedding), top_k=5
        )
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500, detail="Search error occurred"
        )

    # Return the search results
    return {"results": results}
