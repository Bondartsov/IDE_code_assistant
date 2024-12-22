# core/database.py
from core.config import settings
from core.logger import logger

logger.info(f"DATABASE_URL is set to: {settings.DATABASE_URL}")

from typing import Generator
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base, Session

from core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

Base = declarative_base()

class Document(Base):
    """
    Модель документа для хранения в базе данных.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    embedding = Column(String, nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Генератор сессии базы данных.

    Yields:
        Session: Сессия базы данных.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()