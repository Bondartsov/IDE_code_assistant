# services/indexing_service.py

import os
import json
import faiss
import numpy as np

from core.logger import logger

class FAISSIndexManager:
    def __init__(self, save_to_disk=True, index_file='index.faiss', mapping_file='id_to_idx.json'):
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.save_to_disk = save_to_disk
        self.index = faiss.IndexFlatL2(1536)  # Размерность эмбеддинга
        self.id_to_idx = {}
        self.next_idx = 0
        if self.save_to_disk:
            self.load_index()

    def load_index(self):
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

    def save_index(self):
        if not self.save_to_disk:
            return
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'w') as f:
            json.dump(self.id_to_idx, f)

    def add_document(self, doc_id, embedding):
        embedding = np.array([embedding], dtype='float32')
        self.index.add(embedding)
        self.id_to_idx[str(doc_id)] = self.next_idx
        self.next_idx += 1
        self.save_index()

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            doc_id = None
            for key, value in self.id_to_idx.items():
                if value == idx:
                    doc_id = int(key)
                    break
            results.append({'doc_id': doc_id, 'distance': distance})
        return results

    def delete_document(self, doc_id):
        pass  # Deletion not supported in IndexFlatL2

# Создаём глобальный экземпляр index_manager
index_manager = FAISSIndexManager()