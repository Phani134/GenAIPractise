from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import laod_documents as load_documents

""" This Document Embedding module is responsible for splitting documents into smaller chunks
    and generating embeddings for those chunks using a pre-trained model. 
    This document reads the docs loaded by the data_loader module.
"""
class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the EmbeddingPipeline with a specified model and text splitter configuration.

        Args:
            model_name (str): The name of the pre-trained model to use for embeddings.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive text chunks.
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Initializing EmbeddingPipeline with model: {model_name}")
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def generate_embeddings(self, chunks: List[Any]) -> np.ndarray: 
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Generated embeddings for {len(chunks)} chunks.")
        return embeddings

# Example usage:
if __name__ == "__main__":
    
    documents = load_documents()
    embedding_pipeline = EmbeddingPipeline()
    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.generate_embeddings(chunks)
    
    print(f"[INFO] Embedding shape: {embeddings.shape}")