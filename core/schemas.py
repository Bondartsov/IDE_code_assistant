# core/schemas.py

from pydantic import BaseModel, ConfigDict

class UserCreate(BaseModel):
    """
    Схема для создания пользователя.
    """
    email: str
    password: str

class UserRead(BaseModel):
    """
    Схема для чтения информации о пользователе.
    """
    id: int
    email: str

    model_config = ConfigDict(from_attributes=True)