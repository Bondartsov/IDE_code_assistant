# core/test_config.py

from pydantic_settings import BaseSettings

class TestSettings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./test_app.db"  # Укажите путь к тестовой базе данных

test_settings = TestSettings()