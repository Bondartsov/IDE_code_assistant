# tests/test_database.py

import pytest
from sqlalchemy import text
from core.database import SessionLocal

def test_database_connection():
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
    finally:
        db.close()