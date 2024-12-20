# test_api.py

import pytest
from fastapi.testclient import TestClient
from api import app
import os
client = TestClient(app)

def get_api_key():
    """
    Generates and returns an API key for tests.
    """
    response = client.post("/api/generate_key/")
    assert response.status_code == 200
    return response.json()["api_key"]

def test_generate_key():
    """
    Tests the generation of a new API key.
    """
    response = client.post("/api/generate_key/")
    assert response.status_code == 200
    assert "api_key" in response.json()

def test_get_models():
    """
    Tests retrieving the list of OpenAI models.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}
    response = client.get("/api/models/", headers=headers)
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) > 0

def test_run_openai_prompt():
    """
    Tests executing a request to the /api/openai/ endpoint.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}
    json_data = {
        "prompt": "What is the capital of France?"
    }
    response = client.post("/api/openai/", headers=headers, json=json_data)
    assert response.status_code == 200, f"Response status code: {response.status_code}, detail: {response.text}"
    assert "response" in response.json()
    print(response.json()["response"])

def test_api_key_expiry_cleanup():
    """
    Tests the cleanup of expired API keys.
    """
    # Generate a new API key
    api_key = get_api_key()

    # Simulate time passing to expire the API key
    # This might involve mocking the datetime or providing a test endpoint
    # For this example, we'll assume there is an endpoint to expire keys
    # Expire the API key (this endpoint is hypothetical)
    expire_response = client.post(f"/api/expire_key/{api_key}/")
    assert expire_response.status_code == 200

    # Try to use the expired API key
    headers = {"api-key": api_key}
    model_response = client.get("/api/models/", headers=headers)
    assert model_response.status_code == 401  # Unauthorized due to expired key
    
def test_upload_unsupported_file_format():
    """
    Tests uploading a file with an unsupported format.
    """
    api_key = get_api_key()
    headers = {"api-key": api_key}

    # Create a test file with an unsupported format
    file_content = b"This is a test file with unsupported format."
    files = {
        "file": ("test.txt", file_content, "text/plain")
    }
    data = {
        "title": "Unsupported File Test"
    }

    response = client.post(
        "/api/knowledge_base/",
        headers=headers,
        files=files,
        data=data
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type"

def test_upload_supported_pdf_file():
    api_key = get_api_key()
    headers = {
        "api-key": api_key
    }

    # Открываем реальный PDF-файл
    with open("test_files/sample.pdf", "rb") as f:
        pdf_content = f.read()

    files = {
        "file": ("sample.pdf", pdf_content, "application/pdf")
    }
    data = {
        "title": "PDF File Test"
    }

    response = client.post(
        "/api/knowledge_base/",
        headers=headers,
        files=files,
        data=data
    )

    assert response.status_code == 200
    assert response.json()["detail"] == "Document added successfully"

    