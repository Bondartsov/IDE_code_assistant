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
    Класс для управления конфигурацией приложения, включая управление API-ключами и взаимодействие с базой данных.
    """

    def __init__(self):
        # Словарь для хранения валидных API-ключей и времени их истечения
        self.valid_keys = {}
        # Инициализация базы данных
        self.engine = create_engine('sqlite:///app.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

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

    def get_openai_api_key(self):
        """
        Получает API-ключ OpenAI из переменных окружения.

        Возвращает:
            str: API-ключ OpenAI

        Вызывает:
            ValueError: Если API-ключ OpenAI не найден
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API ключ не найден. Добавьте его в .env файл с ключом OPENAI_API_KEY.")
        return api_key

    def add_document(self, title: str, content: str, embedding: bytes):
        """
        Добавляет новый документ в базу данных.

        Аргументы:
            title (str): Заголовок документа
            content (str): Содержимое документа
            embedding (bytes): Векторное представление документа

        Возвращает:
            int: Идентификатор нового документа
        """
        session = self.Session()
        new_doc = Document(title=title, content=content, embedding=embedding)
        session.add(new_doc)
        session.commit()
        doc_id = new_doc.id  # Получаем ID документа после коммита
        session.close()
        return doc_id

    def get_document(self, doc_id: int):
        """
        Получает документ из базы данных по его идентификатору.

        Аргументы:
            doc_id (int): Идентификатор документа

        Возвращает:
            Document: Экземпляр документа или None, если не найден
        """
        session = self.Session()
        doc = session.query(Document).filter(Document.id == doc_id).first()
        session.close()
        return doc

    def delete_document(self, doc_id: int):
        """
        Удаляет документ из базы данных по его идентификатору.

        Аргументы:
            doc_id (int): Идентификатор документа для удаления
        """
        session = self.Session()
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc:
            session.delete(doc)
            session.commit()
        session.close()

    def get_all_documents(self):
        """
        Получает список всех документов из базы данных.

        Возвращает:
            list: Список объектов Document
        """
        session = self.Session()
        documents = session.query(Document).all()
        session.close()
        return documents