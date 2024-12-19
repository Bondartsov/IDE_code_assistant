# Файл: index_manager.py

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from config_storage import ConfigManager  # Предполагаем, что этот модуль существует

class IndexManager:
    """
    Интерфейс для управления индексом векторных представлений.
    """

    def add_document(self, doc_id: int, embedding: np.ndarray, metadata: Dict[str, Any]):
        """
        Добавляет документ в индекс.

        :param doc_id: Уникальный идентификатор документа.
        :param embedding: Векторное представление (эмбеддинг) документа.
        :param metadata: Словарь с метаданными документа.
        """
        pass

    def update_document(self, doc_id: int, embedding: np.ndarray, metadata: Dict[str, Any]):
        """
        Обновляет существующий документ в индексе.

        :param doc_id: Уникальный идентификатор документа.
        :param embedding: Новое векторное представление документа.
        :param metadata: Обновленные метаданные документа.
        """
        pass

    def delete_document(self, doc_id: int):
        """
        Удаляет документ из индекса.

        :param doc_id: Уникальный идентификатор документа.
        """
        pass

    def search(self, embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Ищет наиболее похожие документы в индексе.

        :param embedding: Векторное представление запроса.
        :param top_k: Количество возвращаемых результатов.
        :return: Список найденных документов с их метаданными.
        """
        pass

class FAISSIndexManager(IndexManager):
    def __init__(self, index_file: str = "faiss_index.bin", dimension: int = 1536):
        """
        Инициализирует менеджер индекса FAISS.

        :param index_file: Путь к файлу сохранения индекса.
        :param dimension: Размерность векторных представлений.
        """
        self.index_file = index_file
        self.dimension = dimension
        self.config_manager = ConfigManager()  # Используем ConfigManager для управления метаданными
        self.id_to_idx = {}  # Соответствие doc_id и индекса в FAISS
        if os.path.exists(self.index_file):
            # Загружаем сохраненный индекс
            self.index = faiss.read_index(self.index_file)
            with open('id_to_idx.pkl', 'rb') as f:
                self.id_to_idx = pickle.load(f)
        else:
            # Создаем новый индекс
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_document(self, doc_id: int, embedding: np.ndarray, metadata: Dict[str, Any]):
        embedding = np.array([embedding]).astype('float32')
        self.index.add(embedding)
        self.id_to_idx[doc_id] = self.index.ntotal - 1  # Индекс нового документа
        # Удаляем следующий вызов, так как он не нужен
        # self.config_manager.save_document(doc_id, metadata, embedding.tobytes())
        self.save_index()

    def update_document(self, doc_id: int, embedding: np.ndarray, metadata: Dict[str, Any]):
        self.delete_document(doc_id)
        self.add_document(doc_id, embedding, metadata)

    def delete_document(self, doc_id: int):
        # Удаляем документ из ConfigManager
        self.config_manager.delete_document(doc_id)
        # Переиндексируем все данные
        self.reindex()

    def reindex(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_idx = {}
        all_docs = self.config_manager.get_all_documents()
        embeddings = []
        for idx, doc in enumerate(all_docs):
            embedding = np.frombuffer(doc.embedding, dtype='float32')
            embeddings.append(embedding)
            self.id_to_idx[doc.id] = idx  # Обновляем соответствие doc_id и индекса

        if embeddings:
            embeddings = np.array(embeddings).astype('float32')
            self.index.add(embeddings)

        self.save_index()

    def search(self, embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        embedding = np.array([embedding]).astype('float32')
        distances, indices = self.index.search(embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < self.index.ntotal:
                # Получаем doc_id по индексу
                doc_id = None
                for id_, index in self.id_to_idx.items():
                    if index == idx:
                        doc_id = id_
                        break
                if doc_id is not None:
                    # Получаем данные документа из ConfigManager
                    doc = self.config_manager.get_document(doc_id)
                    results.append({
                        'doc_id': doc.id,
                        'title': doc.title,
                        'content': doc.content
                    })
        return results

    def save_index(self):
        faiss.write_index(self.index, self.index_file)
        with open('id_to_idx.pkl', 'wb') as f:
            pickle.dump(self.id_to_idx, f)
