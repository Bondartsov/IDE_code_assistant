# Файл: config_storage.py

import os
import secrets
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Загружаем переменные окружения из файла .env
load_dotenv()

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

class ConfigManager:
    """
    Класс для управления конфигурацией приложения, включая управление API-ключами,
    настройками API провайдера и взаимодействие с базой данных.
    """

    def __init__(self):
        self.valid_keys = {}
        # Инициализация базы данных
        self.engine = create_engine('sqlite:///app.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Загружаем настройки из .env
        self.api_provider = os.getenv("API_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.lmstudio_api_url = os.getenv("LMSTUDIO_API_URL")

    def generate_api_key(self):
        """
        Генерирует новый API-ключ и сохраняет его со временем истечения.

        Возвращает:
            str: Сгенерированный API-ключ
        """
        new_key = secrets.token_hex(16)
        expiry_time = datetime.now() + timedelta(hours=24)
        self.valid_keys[new_key] = expiry_time
        return new_key

    def validate_api_key(self, key):
        """
        Проверяет валидность предоставленного API-ключа.

        Аргументы:
            key (str): API-ключ для проверки

        Возвращает:
            bool: True, если ключ валиден и не истек, иначе False
        """
        self.cleanup_expired_keys()
        return key in self.valid_keys

    def cleanup_expired_keys(self):
        """
        Удаляет устаревшие API-ключи из списка валидных ключей.
        """
        now = datetime.now()
        self.valid_keys = {k: v for k, v in self.valid_keys.items() if v > now}

    def get_api_provider(self):
        """
        Возвращает выбранного поставщика API.

        Возвращает:
            str: Название поставщика API
        """
        return self.api_provider

    def get_openai_api_key(self):
        """
        Получает API-ключ OpenAI из переменных окружения.

        Возвращает:
            str: API-ключ OpenAI

        Вызывает:
            ValueError: Если API-ключ OpenAI не найден
        """
        if not self.openai_api_key and self.api_provider == "openai":
            raise ValueError("API ключ OpenAI не найден. Добавьте его в файл .env с ключом OPENAI_API_KEY.")
        return self.openai_api_key

    def get_lmstudio_api_url(self):
        """
        Получает URL LMStudio из переменных окружения.

        Возвращает:
            str: URL LMStudio

        Вызывает:
            ValueError: Если URL LMStudio не найден
        """
        if not self.lmstudio_api_url and self.api_provider == "lmstudio":
            raise ValueError("URL LMStudio не найден. Добавьте его в файл .env с ключом LMSTUDIO_API_URL.")
        return self.lmstudio_api_url
    def add_document(self, title: str, content: str, embedding: bytes):
        """
        Добавляет новый документ в базу данных.

        Аргументы:
            title (str): Заголовок документа
            content (str): Содержимое документа
            embedding (bytes): Векторное представление документа

        Возвращает:
            int: ID добавленного документа
        """
        session = self.Session()
        new_doc = Document(title=title, content=content, embedding=embedding)
        session.add(new_doc)
        session.commit()
        doc_id = new_doc.id  # Получаем ID документа после коммита
        session.close()
        return doc_id

    # Добавьте метод get_document
    def get_document(self, doc_id: int):
        """
        Получает документ из базы данных по его ID.

        Аргументы:
            doc_id (int): ID документа

        Возвращает:
            Document: Объект документа или None, если не найден
        """
        session = self.Session()
        document = session.query(Document).filter_by(id=doc_id).first()
        session.close()
        return document