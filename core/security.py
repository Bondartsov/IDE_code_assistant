# core/security.py

from datetime import datetime, timedelta
from typing import Dict

from fastapi import Request, HTTPException, status
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
                # Ключ истек
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

def validate_api_key(request: Request):
    """
    Валидирует API-ключ из заголовков запроса.
    Args:
        request (Request): Запрос клиента.
    Returns:
        str: Проверенный API-ключ.

    Raises:
        HTTPException: Если API-ключ отсутствует или недействителен.
    """
    api_key = request.headers.get("api-key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing"
        )
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key"
        )
    return api_key