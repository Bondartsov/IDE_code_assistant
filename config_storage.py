# File: config_storage.py

import os
import secrets
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from the .env file
load_dotenv()

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    document_id = Column(String(36), nullable=False)  # Изменено на String(36)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

class ConfigManager:
    """
    Class for managing the application's configuration, including API key management,
    API provider settings, and database interactions.
    """

    def __init__(self):
        self.valid_keys = {}
        # Initialize the database
        self.engine = create_engine('sqlite:///app.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Load settings from the .env file
        self.api_provider = os.getenv("API_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.lmstudio_api_url = os.getenv("LMSTUDIO_API_URL")

    def generate_api_key(self):
        """
        Generates a new API key and saves it with an expiration time.

        Returns:
            str: Generated API key
        """
        new_key = secrets.token_hex(16)
        expiry_time = datetime.now() + timedelta(hours=24)
        self.valid_keys[new_key] = expiry_time
        return new_key

    def validate_api_key(self, key):
        """
        Validates the provided API key.

        Args:
            key (str): API key to validate

        Returns:
            bool: True if the key is valid and not expired, else False
        """
        self.cleanup_expired_keys()
        return key in self.valid_keys

    def cleanup_expired_keys(self):
        """
        Removes expired API keys from the list of valid keys.
        """
        now = datetime.now()
        self.valid_keys = {k: v for k, v in self.valid_keys.items() if v > now}

    def get_api_provider(self):
        """
        Returns the selected API provider.

        Returns:
            str: Name of the API provider
        """
        return self.api_provider

    def get_openai_api_key(self):
        """
        Retrieves the OpenAI API key from environment variables.

        Returns:
            str: OpenAI API key

        Raises:
            ValueError: If the OpenAI API key is not found
        """
        if not self.openai_api_key and self.api_provider == "openai":
            raise ValueError("OpenAI API key not found. Add it to the .env file with the key OPENAI_API_KEY.")
        return self.openai_api_key

    def get_lmstudio_api_url(self):
        """
        Retrieves the LMStudio API URL from environment variables.

        Returns:
            str: LMStudio API URL

        Raises:
            ValueError: If the LMStudio API URL is not found
        """
        if not self.lmstudio_api_url and self.api_provider == "lmstudio":
            raise ValueError("LMStudio API URL not found. Add it to the .env file with the key LMSTUDIO_API_URL.")
        return self.lmstudio_api_url

    def add_document(self, title: str, content: str, embedding: bytes, document_id: str):
        """
        Adds a new document to the database.

        Args:
            title (str): Title of the document
            content (str): Content of the document
            embedding (bytes): Embedding vector representation of the document
            document_id (str): Document ID

        Returns:
            int: ID of the added document
        """
        session = self.Session()
        new_doc = Document(
            document_id=document_id,
            title=title,
            content=content,
            embedding=embedding
        )
        session.add(new_doc)
        session.commit()
        doc_id = new_doc.id  # Obtain the document ID after commit
        session.close()
        return doc_id

    def get_document(self, doc_id: int):
        """
        Retrieves a document from the database by its ID.

        Args:
            doc_id (int): ID of the document

        Returns:
            Document: Document object or None if not found
        """
        session = self.Session()
        document = session.query(Document).filter_by(id=doc_id).first()
        session.close()
        return document