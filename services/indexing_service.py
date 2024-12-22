# services/indexing_service.py

"""
Module for managing the FAISS index.
"""

import faiss
import numpy as np
from typing import List
import os
import pickle

from core.logger import logger

class IndexingService:
    """
    Class for managing the FAISS index.
    """
    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        id_map_path: str = "id_to_idx.pkl",
        documents_path: str = "documents.pkl"
    ) -> None:
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.documents_path = documents_path
        self.index = None
        self.id_to_idx = {}
        self.documents = {}
        self.next_id = 0  # Unique ID for documents
        self.load_index()

    def load_index(self) -> None:
        """
        Loads the index and mappings from disk.
        """
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = None
        else:
            self.index = None
        if os.path.exists(self.id_map_path):
            try:
                with open(self.id_map_path, "rb") as f:
                    self.id_to_idx = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load ID to index mapping: {e}")
                self.id_to_idx = {}
        else:
            self.id_to_idx = {}

        if os.path.exists(self.documents_path):
            try:
                with open(self.documents_path, "rb") as f:
                    self.documents = pickle.load(f)
                    if self.documents:
                        self.next_id = max(self.documents.keys()) + 1
            except Exception as e:
                logger.error(f"Failed to load documents: {e}")
                self.documents = {}
        else:
            self.documents = {}

    def save_index(self) -> None:
        """
        Saves the index and mappings to disk.
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_to_idx, f)
        with open(self.documents_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, texts: List[str], embeddings: List[np.ndarray]) -> None:
        """
        Adds multiple documents to the index.
        Args:
            texts (List[str]): List of document texts.
            embeddings (List[np.ndarray]): List of document embeddings.
        """
        embeddings_array = np.vstack(embeddings).astype("float32")
        num_embeddings = embeddings_array.shape[0]

        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        # Update id_to_idx and save documents
        for i in range(num_embeddings):
            doc_id = self.next_id
            self.id_to_idx[doc_id] = self.index.ntotal - num_embeddings + i
            self.documents[doc_id] = {
                "title": f"Document {doc_id}",
                "content": texts[i],
            }
            self.next_id += 1

        self.save_index()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Searches for the most similar documents.
        Args:
            query_embedding (np.ndarray): Embedding of the query.
            top_k (int): Number of results to return.
        Returns:
            List[int]: List of document IDs.
        """
        if self.index is None:
            return []
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        indices = indices.flatten()
        # Get actual document IDs
        doc_ids = []
        for idx in indices:
            doc_id = next((doc_id for doc_id, index in self.id_to_idx.items() if index == idx), None)
            if doc_id is not None:
                doc_ids.append(doc_id)
        return doc_ids

    def get_documents_by_ids(self, ids: List[int]) -> List[dict]:
        """
        Retrieves documents by their IDs.
        Args:
            ids (List[int]): List of document IDs.

        Returns:
            List[dict]: List of documents.
        """
        results = []
        for doc_id in ids:
            doc = self.documents.get(doc_id)
            if doc:
                results.append({
                    "id": doc_id,
                    "title": doc.get("title", f"Document {doc_id}"),
                    "content": doc.get("content", "")
                })
        return results

    def delete_document(self, doc_id: int) -> None:
        """
        Deletes a document from the index.

        Args:
            doc_id (int): Document ID.

        Note:
            Deletion is not supported in IndexFlatL2.
        """
        # Deletion not supported in IndexFlatL2
        pass

# Создаём экземпляр сервиса при импорте
indexing_service = IndexingService()
