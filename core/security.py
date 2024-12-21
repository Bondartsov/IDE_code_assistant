# core/security.py

from datetime import datetime, timedelta
from typing import Dict

class APIKeyManager:
    def __init__(self):
        self.api_keys: Dict[str, datetime] = {}

    def create_api_key(self, expires_in_hours: int = 24) -> str:
        import uuid
        api_key = uuid.uuid4().hex
        self.api_keys[api_key] = datetime.utcnow() + timedelta(hours=expires_in_hours)
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        if api_key in self.api_keys:
            if datetime.utcnow() < self.api_keys[api_key]:
                return True
            else:
                # Ключ истёк
                del self.api_keys[api_key]
        return False

    def invalidate_api_key(self, api_key: str):
        if api_key in self.api_keys:
            del self.api_keys[api_key]

api_key_manager = APIKeyManager()

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="api-key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key_manager.validate_api_key(api_key):
        return api_key
    else:
        raise HTTPException(status_code=401, detail="Invalid API Key")