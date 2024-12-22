# core/security.py

from datetime import datetime, timedelta
from typing import Dict

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

class APIKeyManager:
    """
    Класс для управления API-ключами.
    """
    def __init__(self) -> None:
        self.api_keys: Dict[str, datetime] = {}

    def create_api_key(self, expires_in_hours: int = 24) -> str:
        """
        Создает новый API-ключ.

        Args:
            expires_in_hours (int): Срок действия ключа в часах.

        Returns:
            str: Сгенерированный API-ключ.
        """
        import uuid
        api_key = uuid.uuid4().hex
        self.api_keys[api_key] = datetime.utcnow() + timedelta(hours=expires_in_hours)
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        """
        Проверяет действительность API-ключа.

        Args:
            api_key (str): Проверяемый API-ключ.

        Returns:
            bool: Результат проверки.
        """
        if api_key in self.api_keys:
            if datetime.utcnow() < self.api_keys[api_key]:
                return True
            else:
                # Key has expired
                del self.api_keys[api_key]
        return False

    def invalidate_api_key(self, api_key: str) -> None:
        """
        Аннулирует API-ключ.

        Args:
            api_key (str): API-ключ для аннулирования.
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]

api_key_manager = APIKeyManager()

api_key_header = APIKeyHeader(name="api-key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Зависимость для проверки API-ключа.

    Args:
        api_key (str): API-ключ из заголовка запроса.

    Returns:
        str: Проверенный API-ключ.

    Raises:
        HTTPException: Если API-ключ недействителен.
    """
    if api_key and api_key_manager.validate_api_key(api_key):
        return api_key
    else:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key",
        )