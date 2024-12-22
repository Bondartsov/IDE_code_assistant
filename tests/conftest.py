# tests/conftest.py

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database import Base
from core.config import settings

TEST_DATABASE_URL = "sqlite:///./test_app.db"

@pytest.fixture(scope='session', autouse=True)
def setup_test_database():
    """
    Настраивает тестовую базу данных перед запуском тестов и очищает после.
    """
    # Сохраняем оригинальный DATABASE_URL
    original_database_url = settings.DATABASE_URL
    # Переопределяем DATABASE_URL на тестовую базу данных
    settings.DATABASE_URL = TEST_DATABASE_URL

    # Создаём двигатель и сессию для тестовой базы данных
    engine = create_engine(settings.DATABASE_URL)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Создаём таблицы в тестовой базе данных
    Base.metadata.create_all(bind=engine)

    yield  # Запускаем тесты

    # После выполнения тестов удаляем таблицы и восстанавливаем DATABASE_URL
    Base.metadata.drop_all(bind=engine)
    settings.DATABASE_URL = original_database_url