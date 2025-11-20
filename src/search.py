import os
from dotenv import load_dotenv
from src.vectorstore import VectorStore
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class RAGSearch:
    def __init__(self, persist_dir: str = "vector_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the RAGSearch with a FAISS vector store.

        Args:
            persist_dir (str): Directory to persist the vector store.
            embedding_model (str): The name of the pre-trained model to use for embeddings.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive text chunks.
        """
        self.vectorstore = VectorStore(persist_dir=persist_dir, embedding_model=embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Load existing vector store if available
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()
        else:
            from src.data_loader import laod_documents
            documents = laod_documents("data")
            self.vectorstore.build_from_documents(documents)
        self.llm = ChatGroq(groq_api_key=groq_api_key,model=llm_model)
        print(f"[INFO] Initialized RAGSearch with LLM model: {llm_model}")
    
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Search the vector store for relevant documents and generate a summary using the LLM.

        Args:
            query (str): The search query.
            top_k (int): The number of top relevant documents to retrieve.
        """
        results = self.vectorstore.query(query, top_k=top_k)
        context = "\n\n".join([res['text'] for res in results])
        if not context.strip():
            return "No relevant documents found."
        prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm.invoke(prompt)
        return response.content
    
# Example usage:
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is Langchain?"
    answer = rag_search.search_and_summarize(query, top_k=3)
    print(f"Answer:\n{answer}")


