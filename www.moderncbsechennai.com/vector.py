import os
from typing import Optional, List

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Simple in-memory cache to avoid rebuilding the same file repeatedly
_VECTOR_CACHE = {}

def _get_embeddings():
    return VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1"),
    )

def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _split_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def load_vector_store(file_path: str):
    """
    Build a FAISS retriever from a .txt file. Returns a retriever with sensible defaults.
    Caches per file path in memory to keep startup fast on Cloud Run.
    """
    try:
        abs_path = os.path.abspath(file_path)
        if abs_path in _VECTOR_CACHE:
            return _VECTOR_CACHE[abs_path]

        if not os.path.exists(abs_path):
            print(f"ℹ️ Vector file not found: {abs_path}")
            return None

        text = _read_text(abs_path)
        docs = _split_text(text)
        embeddings = _get_embeddings()

        vs = FAISS.from_documents(docs, embeddings)
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        _VECTOR_CACHE[abs_path] = retriever
        print(f"📦 Built FAISS index for: {abs_path} (chunks={len(docs)})")
        return retriever

    except Exception as e:
        print(f"⚠️ load_vector_store error for {file_path}: {e}")
        return None
