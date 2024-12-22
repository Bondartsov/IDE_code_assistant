# clear_documents.py

from sqlalchemy.orm import Session
from core.database import SessionLocal, Base, engine
from core.database import Document

def clear_all_documents():
    """
    Удаляет все записи из таблицы документов.
    """
    db: Session = SessionLocal()
    try:
        db.query(Document).delete()
        db.commit()
        print("All documents have been deleted from the database.")
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_documents()