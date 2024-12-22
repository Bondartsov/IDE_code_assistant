# clear_documents.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config import settings
from core.database import Base, Document
import os

def clear_all_documents():
    """
    Удаляет все записи из таблицы документов в базе данных и удаляет связанные файлы индекса.
    """
    # Подключаемся к базе данных, используя DATABASE_URL из настроек
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        # Удаляем все записи из таблицы documents
        db.query(Document).delete()
        db.commit()
        print("Все документы удалены из базы данных.")
    except Exception as e:
        db.rollback()
        print(f"Произошла ошибка при удалении документов из базы данных: {e}")
    finally:
        db.close()

    # Список файлов, которые необходимо удалить
    files_to_delete = ['documents.pkl', 'index.faiss', 'id_to_idx.pkl']

    # Удаляем файлы, если они существуют
    for filename in files_to_delete:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Файл {filename} удален.")
            except Exception as e:
                print(f"Не удалось удалить {filename}: {e}")
        else:
            print(f"Файл {filename} не найден.")

if __name__ == "__main__":
    clear_all_documents()