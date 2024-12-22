# tests/test_indexing_service.py

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*numpy.*"
)

import pytest
from services.indexing_service import FAISSIndexManager
import numpy as np

def test_add_and_search_document():
    # Ваш тестовый код

    warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*numpy\\.core.*"
)

def test_add_and_search_document():
    """
    Тест добавления и поиска документа в индексе FAISS.
    """
    index_manager = FAISSIndexManager(save_to_disk=False)
    embedding = np.ones(1536, dtype='float32')
    doc_id = 42

    index_manager.add_document(doc_id, embedding)
    results = index_manager.search(embedding, top_k=1)

    assert len(results) >= 1
    assert any(result['doc_id'] == doc_id for result in results)