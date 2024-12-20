# Файл: core/security.py

from fastapi import Depends, Header, HTTPException
from datetime import datetime, timedelta
import secrets

from core.database import SessionLocal, APIKey
from core.logger import logger

class APIKeyManager:
    """
    Класс для управления API-ключами.
    """
    def __init__(self):
        self.valid_keys = {}
        self.load_api_keys()

    def load_api_keys(self):
        """
        Загрузка существующих API-ключей из базы данных.
        """
        db = SessionLocal()
        api_keys = db.query(APIKey).all()
        for api_key in api_keys:
            self.valid_keys[api_key.key] = api_key.created_at
        db.close()

    def create_api_key(self):
        """
        Создание нового API-ключа и сохранение его в базе данных.
        """
        new_key = secrets.token_hex(16)
        self.valid_keys[new_key] = datetime.utcnow()
        db = SessionLocal()
        api_key = APIKey(key=new_key)
        db.add(api_key)
        db.commit()
        db.close()
        return new_key

    def validate_api_key(self, api_key: str) -> bool:
        """
        Проверка, действителен ли API-ключ.
        """
        return api_key in self.valid_keys

    def invalidate_api_key(self, api_key: str) -> bool:
        """
        Отзыв существующего API-ключа.
        """
        if api_key in self.valid_keys:
            del self.valid_keys[api_key]
            db = SessionLocal()
            api_key_obj = db.query(APIKey).filter(APIKey.key == api_key).first()
            if api_key_obj:
                db.delete(api_key_obj)
                db.commit()
            db.close()
            return True
        return False

api_key_manager = APIKeyManager()

def get_current_api_key(
    api_key: str = Header(..., alias="api-key")
):
    """
    Зависимость для проверки текущего API-ключа.
    """
    if not api_key_manager.validate_api_key(api_key):
        logger.warning(f"Недействительный API-ключ: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key