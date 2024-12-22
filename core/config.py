# core/config.py

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Класс для хранения настроек конфигурации приложения.
    """
    API_PROVIDER: str = Field(default="openai")
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    LMSTUDIO_API_URL: Optional[str] = Field(default=None)
    LMSTUDIO_EMBEDDING_API_URL: Optional[str] = Field(default=None)
    MODEL_NAME: str = Field(default="chatgpt-4")
    DATABASE_URL: str = Field(default="sqlite:///./app.db")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()