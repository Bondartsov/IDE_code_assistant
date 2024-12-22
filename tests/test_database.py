# tests/test_database.py

import pytest
from sqlalchemy import text
from core.database import SessionLocal

def test_database_connection():
    """
    Тестирование подключения к базе данных.
    """
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT 1")).scalar()
        assert result == 1
    finally:
        db.close()