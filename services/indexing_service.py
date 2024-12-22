# services/indexing_service.py

"""
Модуль для управления индексом FAISS.
"""

import os
import json
import faiss
import numpy as np

from core.logger import logger

class FAISSIndexManager:
    """
    Класс для управления индексом FAISS.
    """
    def __init__(
        self,
        save_to_disk: bool = True,
        index_file: str = 'index.faiss',
        mapping_file: str = 'id_to_idx.json'
    ) -> None:
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.save_to_disk = save_to_disk
        self.index = faiss.IndexFlatL2(1536)
        self.id_to_idx = {}
        self.next_idx = 0
        if self.save_to_disk:
            self.load_index()

    def load_index(self) -> None:
        """
        Загружает индекс и отображение из файлов.
        """
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = faiss.IndexFlatL2(1536)
        else:
            self.index = faiss.IndexFlatL2(1536)
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    self.id_to_idx = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load ID to index mapping: {e}")
                self.id_to_idx = {}
        else:
            self.id_to_idx = {}
        self.next_idx = max(self.id_to_idx.values(), default=-1) + 1

    def save_index(self) -> None:
        """
        Сохраняет индекс и отображение на диск.
        """
        if not self.save_to_disk:
            return
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'w') as f:
            json.dump(self.id_to_idx, f)

    def add_document(self, doc_id: int, embedding: list[float]) -> None:
        """
        Добавляет документ в индекс.

        Args:
            doc_id (int): Идентификатор документа.
            embedding (list[float]): Эмбеддинг документа.
        """
        embedding_vector = np.array([embedding], dtype='float32')
        self.index.add(embedding_vector)
        self.id_to_idx[str(doc_id)] = self.next_idx
        self.next_idx += 1
        self.save_index()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Ищет наиболее похожие документы.

        Args:
            query_embedding (list[float]): Эмбеддинг запроса.
            top_k (int): Количество результатов.

        Returns:
            list[dict]: Список найденных документов.
        """
        query_vector = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            doc_id = next((int(key) for key, value in self.id_to_idx.items() if value == idx), None)
            results.append({'doc_id': doc_id, 'distance': float(distance)})
        return results

    def delete_document(self, doc_id: int) -> None:
        """
        Удаляет документ из индекса.

        Args:
            doc_id (int): Идентификатор документа.

        Note:
            Удаление не поддерживается в IndexFlatL2.
        """
        # Deletion not supported in IndexFlatL2
        pass

# Создаём глобальный экземпляр index_manager
index_manager = FAISSIndexManager()