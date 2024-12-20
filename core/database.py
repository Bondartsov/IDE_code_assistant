# Файл: core/database.py

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Путь к базе данных SQLite
DATABASE_URL = "sqlite:///./app.db"

# Создание движка базы данных
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Создание сессии базы данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для моделей
Base = declarative_base()

def get_db():
    """
    Генератор сессий базы данных для использования в зависимостях.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Document(Base):
    """
    Модель для хранения документов.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(36), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

class APIKey(Base):
    """
    Модель для хранения API-ключей.
    """
    __tablename__ = "api_keys"

    key = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Создание таблиц в базе данных
Base.metadata.create_all(bind=engine)