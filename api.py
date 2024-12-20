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
import requests  # Added for interaction with LMStudio
import tiktoken

# Import ConfigManager and FAISSIndexManager from their respective files
from config_storage import ConfigManager
from index_manager import FAISSIndexManager

# Инициализируем приложение и менеджеры
app = FastAPI()
config_manager = ConfigManager()
index_manager = FAISSIndexManager(config_manager)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Data models for requests
class OpenAIRequest(BaseModel):
    prompt: str

class SearchRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    title: str
    content: str = None

# Function to count the number of tokens in a string
def num_tokens_from_string(string: str, encoding_name: str = "gpt2") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to generate embeddings
def generate_embedding(text: str):
    api_provider = config_manager.get_api_provider()
    if api_provider == "openai":
        # Use OpenAI to generate embeddings
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            return embedding
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAIError: {e}")
            raise HTTPException(
                status_code=500, detail=f"OpenAI API error: {str(e)}"
            )
    elif api_provider == "lmstudio":
        # Use LMStudio to generate embeddings
        lmstudio_url = config_manager.get_lmstudio_api_url()
        try:
            response = requests.post(
                f"{lmstudio_url}/embedding",
                json={"text": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not embedding:
                raise ValueError("Embedding not found in LMStudio response")
            return embedding
        except Exception as e:
            logging.error(f"LMStudio error: {e}")
            raise HTTPException(
                status_code=500, detail=f"LMStudio API error: {str(e)}"
            )
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")

# Function to generate responses
def generate_response(prompt: str):
    api_provider = config_manager.get_api_provider()
    if api_provider == "openai":
        # Use OpenAI to generate the response
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content
            return response_text
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAIError: {e}")
            raise HTTPException(
                status_code=500, detail=f"OpenAI API error: {str(e)}"
            )
    elif api_provider == "lmstudio":
        # Use LMStudio to generate the response
        lmstudio_url = config_manager.get_lmstudio_api_url()
        try:
            response = requests.post(
                f"{lmstudio_url}/chat",
                json={"prompt": prompt}
            )
            response.raise_for_status()
            response_text = response.json().get("response")
            if not response_text:
                raise ValueError("Response not found in LMStudio response")
            return response_text
        except Exception as e:
            logging.error(f"LMStudio error: {e}")
            raise HTTPException(
                status_code=500, detail=f"LMStudio API error: {str(e)}"
            )
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")

# Function to summarize text
def summarize_text(text: str):
    api_provider = config_manager.get_api_provider()
    if api_provider == "openai":
        # Use OpenAI API for summarization
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key
        summary_prompt = f"Please summarize the following text:\n\n{text}"
        try:
            summary_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1024
            )
            summarized_text = summary_response.choices[0].message.content
            return summarized_text
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAIError during summarization: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error during summarization: {str(e)}"
            )
    elif api_provider == "lmstudio":
        # Use LMStudio API for summarization
        lmstudio_url = config_manager.get_lmstudio_api_url()
        try:
            response = requests.post(
                f"{lmstudio_url}/summarize",
                json={"text": text}
            )
            response.raise_for_status()
            summarized_text = response.json().get("summary")
            if not summarized_text:
                raise ValueError("Summary not found in LMStudio response")
            return summarized_text
        except Exception as e:
            logging.error(f"LMStudio error during summarization: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"LMStudio API error during summarization: {str(e)}"
            )
    else:
        raise HTTPException(status_code=500, detail="Unsupported API provider")

# Endpoint to generate a new internal API key
@app.post("/api/generate_key/")
def generate_api_key():
    new_key = config_manager.generate_api_key()
    # Do not log the generated key for security reasons
    logging.info("Generated new API key")
    return {"api_key": new_key}

# Endpoint to revoke an existing API key
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
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key
        # Get the list of models from OpenAI API
        response = openai.Model.list()
        models = [model["id"] for model in response["data"]]
        return {"models": models}
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAIError: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")

# Endpoint to perform a prompt to the model
@app.post("/api/openai/")
def run_openai_prompt(
    request: OpenAIRequest, api_key: str = Header(..., alias="api-key")
):
    # Validate the internal API key
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    prompt = request.prompt

    # Generate embedding for the prompt
    try:
        query_embedding = generate_embedding(prompt)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}"
        )

    # Search for relevant documents
    try:
        relevant_docs = index_manager.search(np.array(query_embedding), top_k=5)
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search error occurred")

    # Build context from found documents
    context = ""
    if relevant_docs:
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    else:
        # If no relevant documents, use the original prompt
        final_prompt = prompt

    # Check the length of the prompt and shorten if necessary
    MAX_TOKENS = 4096
    total_tokens_estimated = num_tokens_from_string(final_prompt)
    if total_tokens_estimated > MAX_TOKENS:
        # Summarize the context if it exists
        if context:
            try:
                # Summarize using summarize_text
                summarized_context = summarize_text(context)
                final_prompt = f"Context:\n{summarized_context}\n\nQuestion:\n{prompt}"
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during summarization: {str(e)}",
                )
        else:
            # If no context, use the original prompt
            pass  # Do nothing

    try:
        # Generate response using generate_response
        response_text = generate_response(final_prompt)
        return {"response": response_text}
    except Exception as e:
        logging.error(f"Error during completion: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during completion: {str(e)}"
        )

# Function to extract text from PDF using pypdf
async def extract_text_from_pdf(file: UploadFile) -> str:
    file.file.seek(0)
    reader = pypdf.PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX using python-docx
async def extract_text_from_docx(file: UploadFile) -> str:
    file.file.seek(0)
    document = docx.Document(file.file)
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
    # Validate the internal API key
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if content is None and file is None:
        raise HTTPException(
            status_code=400, detail="Either content or file must be provided"
        )

    if file is not None:
        # Process the uploaded file
        if file.content_type == "application/pdf":
            content = await extract_text_from_pdf(file)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = await extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Generate embedding for the content
    embedding = generate_embedding(content)

    # Save the document and embedding
    embedding_bytes = np.array(embedding, dtype="float32").tobytes()
    doc_id = config_manager.add_document(title, content, embedding_bytes)
    index_manager.add_document(doc_id, np.array(embedding), {"title": title})

    return {"detail": "Document added successfully", "doc_id": doc_id}

# Endpoint to search the knowledge base
@app.post("/api/search/")
def search_endpoint(
    request: SearchRequest, api_key: str = Header(..., alias="api-key")
):
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    query = request.query

    # Generate embedding for the query
    query_embedding = generate_embedding(query)

    # Perform search in the index
    results = index_manager.search(np.array(query_embedding), top_k=5)

    # Formulate the response
    return {"results": results}