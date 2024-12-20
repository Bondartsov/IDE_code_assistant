# index_manager.py

import faiss
import numpy as np
import pickle
from typing import Dict, Any, List
from config_storage import ConfigManager

class FAISSIndexManager:
    """
    Класс для управления индексом FAISS, используемым для быстрого поиска по эмбеддингам.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Инициализирует менеджер индекса FAISS.

        :param config_manager: Экземпляр ConfigManager для доступа к базе данных.
        """
        self.config_manager = config_manager  # Используем переданный экземпляр ConfigManager
        # Инициализация индекса и загрузка данных
        self.index = faiss.IndexFlatL2(1536)  # Размерность эмбеддинга (1536 для 'text-embedding-ada-002')
        self.id_to_idx = {}  # Соответствие между doc_id и индексом в FAISS
        self.load_index()

    def load_index(self):
        """
        Загружает индекс и соответствия doc_id к индексу в FAISS из файлов.
        """
        try:
            self.index = faiss.read_index("faiss_index.bin")
            with open("id_to_idx.pkl", "rb") as f:
                self.id_to_idx = pickle.load(f)
        except Exception as e:
            # Если файлов нет или возникла ошибка при загрузке, инициализируем пустой индекс
            print(f"Ошибка при загрузке индекса: {e}")
            pass

    def save_index(self):
        """
        Сохраняет индекс и соответствия doc_id к индексу в FAISS в файлы.
        """
        faiss.write_index(self.index, "faiss_index.bin")
        with open("id_to_idx.pkl", "wb") as f:
            pickle.dump(self.id_to_idx, f)

    def add_document(self, doc_id: int, embedding: np.ndarray, metadata: Dict[str, Any]):
        """
        Добавляет документ в индекс.

        :param doc_id: Уникальный идентификатор документа (целое число, первичный ключ из базы данных).
        :param embedding: Векторное представление документа.
        :param metadata: Метаданные документа (не используются в этой версии).
        """
        embedding = np.array([embedding]).astype('float32')
        self.index.add(embedding)
        self.id_to_idx[doc_id] = self.index.ntotal - 1  # Индекс нового документа в FAISS
        self.save_index()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Ищет наиболее похожие документы в индексе.

        :param query_embedding: Векторное представление запроса.
        :param top_k: Количество возвращаемых результатов.
        :return: Список найденных документов с их метаданными.
        """
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        results = []
        for idx in I[0]:
            # Находим doc_id по индексу
            doc_id = None
            for key, value in self.id_to_idx.items():
                if value == idx:
                    doc_id = key
                    break
            if doc_id is not None:
                # Используем self.config_manager для доступа к методу get_document
                document = self.config_manager.get_document(doc_id)
                if document:
                    results.append({
                        "doc_id": document.id,
                        "title": document.title,
                        "content": document.content,
                        "document_id": document.document_id  # Добавляем идентификатор документа
                    })
        return results