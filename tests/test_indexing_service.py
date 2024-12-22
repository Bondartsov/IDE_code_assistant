# tests/test_indexing_service.py

import pytest
from services.indexing_service import IndexingService
import numpy as np
import os

def test_add_and_search_document():
    """
    Тест добавления и поиска документа в индексе FAISS.
    """
    # Создаем экземпляр IndexingService с временными файлами
    index_service = IndexingService(
        index_path="temp_faiss_index.bin",
        id_map_path="temp_id_to_idx.pkl",
        documents_path="temp_documents.pkl"
    )

    # Удаляем временные файлы перед тестом
    for path in ["temp_faiss_index.bin", "temp_id_to_idx.pkl", "temp_documents.pkl"]:
        if os.path.exists(path):
            os.remove(path)

    embedding_dim = 128  # Например, размерность эмбеддинга 128
    embedding = np.ones(embedding_dim, dtype='float32')
    texts = ["Test document"]
    index_service.add_documents(texts, [embedding])

    query_embedding = np.ones(embedding_dim, dtype='float32')
    results = index_service.search(query_embedding, top_k=1)

    assert len(results) >= 1

    retrieved_docs = index_service.get_documents_by_ids(results)
    assert len(retrieved_docs) == 1
    assert retrieved_docs[0]["content"] == "Test document"

    # Удаляем временные файлы после теста
    for path in ["temp_faiss_index.bin", "temp_id_to_idx.pkl", "temp_documents.pkl"]:
        if os.path.exists(path):
            os.remove(path)
