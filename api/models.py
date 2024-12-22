# api/models.py

from pydantic import BaseModel, ConfigDict

class DocumentBase(BaseModel):
    """
    Базовая модель документа.
    """
    title: str
    content: str

class DocumentCreate(DocumentBase):
    """
    Модель для создания документа.
    """
    pass

class Document(DocumentBase):
    """
    Модель документа с идентификатором.
    """
    id: int

    model_config = ConfigDict(from_attributes=True)

class OpenAIRequestModel(BaseModel):
    """
    Модель запроса к OpenAI.
    """
    prompt: str