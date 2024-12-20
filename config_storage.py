# config_storage.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import secrets
import logging

# Загружаем переменные окружения из файла .env
load_dotenv()

# Создаём базовый класс для моделей SQLAlchemy
Base = declarative_base()

class Document(Base):
    """
    Модель для хранения документов в базе данных.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)  # Первичный ключ (автоинкрементный)
    document_id = Column(String(36), nullable=False)  # Уникальный идентификатор документа (UUID)
    title = Column(String, nullable=False)  # Заголовок документа
    content = Column(Text, nullable=False)  # Содержание документа
    embedding = Column(LargeBinary, nullable=False)  # Эмбеддинг документа в двоичном формате
    added_at = Column(DateTime, default=datetime.utcnow)  # Дата и время добавления документа

class APIKey(Base):
    """
    Модель для хранения API-ключей в базе данных.
    """
    __tablename__ = 'api_keys'

    key = Column(String, primary_key=True)  # API-ключ (используется в качестве первичного ключа)
    created_at = Column(DateTime, default=datetime.utcnow)  # Дата и время создания ключа

class ConfigManager:
    """
    Класс для управления конфигурацией приложения и взаимодействия с базой данных.
    """
    def __init__(self):
        # Загружаем конфигурационные параметры из переменных окружения
        self.api_provider = os.getenv("API_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.lmstudio_api_url = os.getenv("LMSTUDIO_API_URL")
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.valid_keys = {}  # Словарь для хранения действительных API-ключей

        # Инициализация базы данных
        self.engine = create_engine('sqlite:///app.db')  # Создаём подключение к базе данных SQLite
        Base.metadata.create_all(self.engine)  # Создаём таблицы, если они ещё не существуют
        self.Session = sessionmaker(bind=self.engine)  # Создаём класс для сессий

        # Загрузка существующих API-ключей из базы данных
        self.load_api_keys()

    def get_api_provider(self):
        """
        Возвращает выбранного провайдера API.
        """
        return self.api_provider

    def get_openai_api_key(self):
        """
        Возвращает API-ключ OpenAI.
        """
        return self.openai_api_key

    def get_lmstudio_api_url(self):
        """
        Возвращает URL LMStudio API.
        """
        return self.lmstudio_api_url

    def get_model_name(self):
        """
        Возвращает имя модели, выбранной для генерации ответов.
        """
        return self.model_name

    def validate_api_key(self, api_key: str) -> bool:
        """
        Проверяет, является ли переданный API-ключ действительным.
        """
        return api_key in self.valid_keys

    def generate_api_key(self):
        """
        Генерирует новый API-ключ и сохраняет его в базе данных.
        """
        new_key = secrets.token_hex(16)  # Генерируем безопасный случайный ключ
        self.valid_keys[new_key] = datetime.utcnow()  # Добавляем ключ в словарь действительных ключей
        # Сохраняем ключ в базе данных
        session = self.Session()
        api_key = APIKey(key=new_key)
        session.add(api_key)
        session.commit()
        session.close()
        return new_key

    def load_api_keys(self):
        """
        Загружает все существующие API-ключи из базы данных.
        """
        session = self.Session()
        api_keys = session.query(APIKey).all()
        for api_key in api_keys:
            self.valid_keys[api_key.key] = api_key.created_at
        session.close()

    def add_document(self, title: str, content: str, embedding_bytes: bytes, document_id: str):
        """
        Добавляет документ в базу данных.

        :param title: Заголовок документа.
        :param content: Содержание документа.
        :param embedding_bytes: Эмбеддинг документа в двоичном формате.
        :param document_id: Уникальный идентификатор документа (UUID).
        :return: ID добавленного документа.
        """
        session = self.Session()
        document = Document(
            document_id=document_id,
            title=title,
            content=content,
            embedding=embedding_bytes
        )
        session.add(document)
        session.commit()
        doc_id = document.id  # Получаем автоинкрементное значение ID документа
        session.close()
        return doc_id

    def get_document(self, doc_id: int):
        """
        Получает документ из базы данных по его ID.

        :param doc_id: Первичный ключ документа (целое число).
        :return: Экземпляр Document или None.
        """
        session = self.Session()
        document = session.query(Document).filter(Document.id == doc_id).first()
        session.close()
        return document

    # Здесь вы можете добавить дополнительные методы по необходимости