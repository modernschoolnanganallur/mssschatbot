import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Cache to avoid rebuilding on every request
_VECTOR_CACHE = {}

DATA_DIR = "data"
VECTOR_STORE_PATH = "faiss_index"

def _get_embeddings():
    """Return OpenAI embedding model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def _read_text(file_path: str) -> str:
    """Read text file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def _split_text(text: str) -> List[Document]:
    """Split large text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def load_all_files(data_dir: str = DATA_DIR) -> List[Document]:
    """Load and split all .txt files in the data directory."""
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            path = os.path.join(data_dir, file)
            try:
                text = _read_text(path)
                docs.extend(_split_text(text))
                print(f"✅ Loaded {file}")
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
    return docs

def load_vector_store():
    """Build or load FAISS retriever from all data files."""
    global _VECTOR_CACHE

    if _VECTOR_CACHE.get("retriever"):
        return _VECTOR_CACHE["retriever"]

    try:
        if os.path.exists(VECTOR_STORE_PATH):
            print("📂 Loading existing FAISS index...")
            embeddings = _get_embeddings()
            vs = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("🧠 Building new FAISS index from /data...")
            docs = load_all_files()
            embeddings = _get_embeddings()
            vs = FAISS.from_documents(docs, embeddings)
            vs.save_local(VECTOR_STORE_PATH)

        retriever = vs.as_retriever(search_kwargs={"k": 4})
        _VECTOR_CACHE["retriever"] = retriever
        print(f"✅ Vector store ready. Total docs: {len(vs.index_to_docstore_id)}")
        return retriever

    except Exception as e:
        print(f"⚠️ load_vector_store error: {e}")
        return None

if __name__ == "__main__":
    load_vector_store()
