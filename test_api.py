# test_api.py

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from api import app
from config_storage import ConfigManager

# Create a test client for the FastAPI application
client = TestClient(app)
config_manager = ConfigManager()

@pytest.fixture
def api_key():
    """
    Fixture for generating and providing an API key for tests.
    """
    response = client.post("/api/generate_key/")
    return response.json()["api_key"]

def test_generate_key():
    """
    Tests the generation of a new API key.
    """
    response = client.post("/api/generate_key/")
    assert response.status_code == 200
    assert "api_key" in response.json()

def test_get_models(api_key):
    """
    Tests retrieving the list of OpenAI models.
    """
    headers = {"api-key": api_key}
    response = client.get("/api/models/", headers=headers)
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) > 0

def test_run_openai_prompt(api_key):
    """
    Tests executing a request to the /api/openai/ endpoint.
    """
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
    config_manager = ConfigManager()
    key = config_manager.generate_api_key()
    # Set an expired time for the key
    config_manager.valid_keys[key] = datetime.now() - timedelta(hours=1)
    config_manager.cleanup_expired_keys()
    assert not config_manager.valid_keys
