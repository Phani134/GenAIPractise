import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from src.embeddings import EmbeddingPipeline
from sentence_transformers import SentenceTransformer

class VectorStore: 
    def __init__(self, persist_dir: str = "vector_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index=None
        self.metadata=[]
        self.model= SentenceTransformer(embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents.")
        embedding_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = embedding_pipeline.chunk_documents(documents)
        embeddings = embedding_pipeline.generate_embeddings(chunks)
        metadatas = [{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype(np.float32), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}.")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to the vector store.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index to {faiss_path} and metadata to {meta_path}.")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded FAISS index from {faiss_path} and metadata from {meta_path}.")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Any]:
        if self.index is None:
            raise ValueError("The vector store is not loaded. Please load or build the index first.")
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx]
            results.append({"metadata": meta, "distance": dist,"index": idx})
        print(f"[INFO] Retrieved top {top_k} results for the query.")
        return results
    
    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        query_embedding = self.model.encode([query_text]).astype(np.float32)
        return self.search(query_embedding, top_k=top_k)
    
# Example usage:
if __name__ == "__main__":
    from src.data_loader import laod_documents
    documents = laod_documents("data")
    store = VectorStore()
    store.build_from_documents(documents)
    store.load()
    query = "Sample query text"
    results = store.query(query, top_k=3)
    print(f"[INFO] Query results: {results}")
