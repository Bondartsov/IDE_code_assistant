# Файл: services/indexing_service.py

import faiss
import numpy as np
import pickle
from typing import Dict, Any, List

from core.logger import logger
from core.database import SessionLocal, Document

class FAISSIndexManager:
    """
    Класс для управления индексом FAISS.
    """
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)  # Размерность эмбеддинга
        self.id_to_idx = {}  # Соответствие между doc_id и индексом в FAISS
        self.load_index()

    def load_index(self):
        """
        Загрузка индекса из файлов.
        """
        try:
            self.index = faiss.read_index("faiss_index.bin")
            with open("id_to_idx.pkl", "rb") as f:
                self.id_to_idx = pickle.load(f)
            logger.info("Индекс FAISS успешно загружен")
        except Exception as e:
            logger.warning(f"Не удалось загрузить индекс: {e}")

    def save_index(self):
        """
        Сохранение индекса в файлы.
        """
        faiss.write_index(self.index, "faiss_index.bin")
        with open("id_to_idx.pkl", "wb") as f:
            pickle.dump(self.id_to_idx, f)

    def add_document(self, doc_id: int, embedding: np.ndarray):
        """
        Добавление документа в индекс.
        """
        embedding = np.array([embedding]).astype('float32')
        self.index.add(embedding)
        self.id_to_idx[doc_id] = self.index.ntotal - 1
        self.save_index()
        logger.info(f"Документ {doc_id} добавлен в индекс")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск наиболее похожих документов по эмбеддингу запроса.
        """
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        results = []
        db = SessionLocal()
        for idx in I[0]:
            # Найти doc_id по индексу
            doc_id = None
            for key, value in self.id_to_idx.items():
                if value == idx:
                    doc_id = key
                    break
            if doc_id is not None:
                document = db.query(Document).filter(Document.id == doc_id).first()
                if document:
                    results.append({
                        "doc_id": document.id,
                        "title": document.title,
                        "content": document.content,
                        "document_id": document.document_id
                    })
        db.close()
        return results

# Экземпляр менеджера индекса
index_manager = FAISSIndexManager()