# api/models.py

from pydantic import BaseModel, ConfigDict

class DocumentBase(BaseModel):
    title: str
    content: str

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: int

    model_config = ConfigDict(from_attributes=True)

class OpenAIRequestModel(BaseModel):
    prompt: str