from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader,TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader

def laod_documents(data_dir:str) -> List[Any]:
    """
    Load documents from the specified directory.

    Args:
        data_dir (str): The directory containing the documents.
    """
    data_path=Path(data_dir).resolve()
    print(f"[DEBUG] Loading documents from: {data_path}")
    documents=[]

    #PDF Files
    pdf_files = list(data_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF file: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            print(f"[DEBUG] Loaded PDF file: {pdf_file}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF file {pdf_file}: {e}")
    print(f"[DEBUG] Total PDF files loaded: {len(pdf_files)}")

    #CSV Files
    csv_files = list(data_path.glob("*.csv"))
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV file: {csv_file}")
        try:
            loader = CSVLoader(file_path=str(csv_file), encoding="utf-8")
            docs = loader.load()
            print(f"[DEBUG] Loaded CSV file: {csv_file}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV file {csv_file}: {e}")
    print(f"[DEBUG] Total CSV files loaded: {len(csv_files)}")

    #txt Files
    txt_files = list(data_path.glob("*.txt"))
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT file: {txt_file}")
        try:
            loader = TextLoader(file_path=str(txt_file), encoding="utf-8")
            docs = loader.load()
            print(f"[DEBUG] Loaded TXT file: {txt_file}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT file {txt_file}: {e}")
    print(f"[DEBUG] Total TXT files loaded: {len(txt_files)}")

    # Word Files
    word_files = list(data_path.glob("*.docx"))
    for word_file in word_files:
        print(f"[DEBUG] Loading Word file: {word_file}")
        try:
            loader = Docx2txtLoader(str(word_file))
            docs = loader.load()
            print(f"[DEBUG] Loaded Word file: {word_file}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load Word file {word_file}: {e}")
    print(f"[DEBUG] Total Word files loaded: {len(word_files)}")

    # JSON Files
    json_files = list(data_path.glob("*.json"))
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON file: {json_file}")
        try:
            loader = JSONLoader(file_path=str(json_file), encoding="utf-8")
            docs = loader.load()
            print(f"[DEBUG] Loaded JSON file: {json_file}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {json_file}: {e}")
    print(f"[DEBUG] Total JSON files loaded: {len(json_files)}")

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents

if __name__ == "__main__":
    # Example usage
    data_directory = "./data"
    loaded_documents = laod_documents(data_directory)
    print(f"Number of documents loaded: {len(loaded_documents)}")