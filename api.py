# File: api.py

from uuid import uuid4  # Добавлен импорт str(uuid4())
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
import requests  # Added for interaction with LMStudio
import tiktoken

# Import ConfigManager and FAISSIndexManager from their respective files
from config_storage import ConfigManager
from index_manager import FAISSIndexManager

# Initialize the application and managers
app = FastAPI()
config_manager = ConfigManager()
index_manager = FAISSIndexManager(config_manager)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Maximum text length for the model
MAX_TEXT_LENGTH = 8191  # Maximum number of tokens for the model
# Maximum file size for uploads
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Function to count the number of tokens in text
def num_tokens(text: str, model_name: str = "text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# Function to split text into chunks of max_tokens
def split_text(
    text: str, max_tokens: int, model_name: str = "text-embedding-ada-002"
) -> List[str]:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# Data models for requests
class OpenAIRequest(BaseModel):
    prompt: str

class SearchRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    title: str
    content: str = None

# Function to generate embeddings
def generate_embedding(text: str):
    api_provider = config_manager.get_api_provider()
    if api_provider == "openai":
        # Use OpenAI to generate embeddings
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key

        # Split text into chunks if necessary
        tokens = num_tokens(text)
        if tokens > MAX_TEXT_LENGTH:
            texts = split_text(text, MAX_TEXT_LENGTH)
        else:
            texts = [text]

        embeddings = []
        for chunk in texts:
            try:
                response = openai.Embedding.create(
                    input=chunk, model="text-embedding-ada-002"
                )
                embedding = response["data"][0]["embedding"]
                embeddings.append(embedding)
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAIError: {e}")
                raise HTTPException(
                    status_code=500, detail=f"OpenAI API error: {str(e)}"
                )
        # Average the embeddings if multiple chunks
        embedding = np.mean(embeddings, axis=0).tolist()
        return embedding

    elif api_provider == "lmstudio":
        # Use LMStudio to generate embeddings
        lmstudio_url = config_manager.get_lmstudio_api_url()

        # Similar handling for LMStudio if needed
        tokens = num_tokens(text)
        if tokens > MAX_TEXT_LENGTH:
            texts = split_text(text, MAX_TEXT_LENGTH)
        else:
            texts = [text]

        embeddings = []
        for chunk in texts:
            try:
                response = requests.post(
                    f"{lmstudio_url}/embedding", json={"text": chunk}
                )
                response.raise_for_status()
                chunk_embedding = response.json().get("embedding")
                if not chunk_embedding:
                    raise ValueError("Embedding not found in LMStudio response")
                embeddings.append(chunk_embedding)
            except Exception as e:
                logging.error(f"LMStudio error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"LMStudio API error: {str(e)}"
                )
        # Average the embeddings if multiple chunks
        embedding = np.mean(embeddings, axis=0).tolist()
        return embedding

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
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content
            return response_text
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAIError: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    elif api_provider == "lmstudio":
        # Use LMStudio to generate the response
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

# Function to summarize text
def summarize_text(text: str):
    api_provider = config_manager.get_api_provider()
    if api_provider == "openai":
        # Use OpenAI API for summarization
        openai_api_key = config_manager.get_openai_api_key()
        openai.api_key = openai_api_key

        # Split text into chunks if necessary
        tokens = num_tokens(text, model_name="gpt-3.5-turbo")
        if tokens > MAX_TEXT_LENGTH:
            texts = split_text(text, MAX_TEXT_LENGTH, model_name="gpt-3.5-turbo")
        else:
            texts = [text]

        summarized_texts = []
        for chunk in texts:
            summary_prompt = f"Please summarize the following text:\n\n{chunk}"
            try:
                summary_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=1024,
                )
                summarized_chunk = summary_response.choices[0].message.content
                summarized_texts.append(summarized_chunk)
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAIError during summarization: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API error during summarization: {str(e)}",
                )
        # Combine the summarized chunks
        full_summary = "\n".join(summarized_texts)
        return full_summary

    elif api_provider == "lmstudio":
        # Use LMStudio API for summarization
        lmstudio_url = config_manager.get_lmstudio_api_url()

        # Similar handling for LMStudio if needed
        tokens = num_tokens(text)
        if tokens > MAX_TEXT_LENGTH:
            texts = split_text(text, MAX_TEXT_LENGTH)
        else:
            texts = [text]

        summarized_texts = []
        for chunk in texts:
            try:
                response = requests.post(
                    f"{lmstudio_url}/summarize", json={"text": chunk}
                )
                response.raise_for_status()
                summarized_chunk = response.json().get("summary")
                if not summarized_chunk:
                    raise ValueError("Summary not found in LMStudio response")
                summarized_texts.append(summarized_chunk)
            except Exception as e:
                logging.error(f"LMStudio error during summarization: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"LMStudio API error during summarization: {str(e)}",
                )
        # Combine the summarized chunks
        full_summary = "\n".join(summarized_texts)
        return full_summary
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
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    prompt = request.prompt

    # Generate embedding for the prompt
    query_embedding = generate_embedding(prompt)

    # Search in the index
    results = index_manager.search(np.array(query_embedding), top_k=5)

    # Form context
    context = ""
    if results:
        # Sort results by doc_id (or other method if needed)
        results.sort(key=lambda x: x["doc_id"])
        context = "\n\n".join([doc["content"] for doc in results])
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    else:
        final_prompt = prompt

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
    contents = await file.read()
    reader = pypdf.PdfReader(BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX using python-docx
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
    file: UploadFile = File(None),
):
    if not config_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if content is None and file is None:
        raise HTTPException(
            status_code=400, detail="Either content or file must be provided"
        )

    if file is not None:
        # Check the file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, detail="File size exceeds the maximum limit."
            )
        # Reset the file pointer after reading
        file.file.seek(0)
        # Process the file
        if file.content_type == "application/pdf":
            content = await extract_text_from_pdf(file)
        elif (
            file.content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            content = await extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    # Check the number of tokens in the text
    tokens = num_tokens(content)
    if tokens > MAX_TEXT_LENGTH:
        # If the text is larger than the maximum size, split it into parts
        content_chunks = split_text(content, MAX_TEXT_LENGTH)
    else:
        content_chunks = [content]

    # Generate a unique identifier for the entire document
    document_id = str(uuid4())  # Изменено на строковое представление UUID

    # Add each part to the database and index
    doc_ids = []
    for idx, chunk in enumerate(content_chunks):
        chunk_title = f"{title} (Part {idx+1})" if len(content_chunks) > 1 else title
        # Generate embedding for the chunk
        embedding = generate_embedding(chunk)
        embedding_bytes = np.array(embedding, dtype="float32").tobytes()
        # Save the chunk in the database
        doc_id = config_manager.add_document(
            chunk_title, chunk, embedding_bytes, document_id=document_id
        )
        # Add to the index
        index_manager.add_document(
            doc_id,
            np.array(embedding),
            {"title": chunk_title, "content": chunk}
        )
        doc_ids.append(doc_id)

    return {"detail": "Document added successfully", "doc_ids": doc_ids}

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

    # Search in the index
    results = index_manager.search(np.array(query_embedding), top_k=5)

    # Group results by document_id
    grouped_results = {}
    for result in results:
        doc_id = result.get("document_id")
        if doc_id not in grouped_results:
            grouped_results[doc_id] = {"title": result["title"], "content": ""}
        grouped_results[doc_id]["content"] += result["content"] + "\n\n"

    # Convert to list
    final_results = list(grouped_results.values())

    # Form the response
    return {"results": final_results}