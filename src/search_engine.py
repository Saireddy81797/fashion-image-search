import faiss
import numpy as np

class FashionSearchEngine:
    def __init__(self, dim=512, index_path=None):
        self.index = faiss.IndexFlatL2(dim)
        if index_path:
            faiss.read_index(index_path)

    def add_embeddings(self, embeddings):
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), top_k
        )
        return distances, indices
