# tests/test_security.py

from core.security import api_key_manager

def test_api_key_creation_and_validation():
    """
    Тест создания и проверки API-ключа.
    """
    api_key = api_key_manager.create_api_key()
    assert api_key_manager.validate_api_key(api_key) is True

    api_key_manager.invalidate_api_key(api_key)
    assert api_key_manager.validate_api_key(api_key) is False