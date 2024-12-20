# Файл: api/models.py

from pydantic import BaseModel

class DocumentBase(BaseModel):
    title: str
    content: str