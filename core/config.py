# Файл: core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Класс для хранения настроек конфигурации приложения.
    """
    API_PROVIDER: str = os.getenv("API_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LMSTUDIO_API_URL: str = os.getenv("LMSTUDIO_API_URL")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "chatgpt-4o-latest")

settings = Settings()