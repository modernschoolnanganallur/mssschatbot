# api.py
import os
import json
import logging
import glob
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# OpenAI client
from openai import OpenAI

# LangChain/FAISS pieces
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np

# -----------------------
# Basic configuration
# -----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("msss")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY must be set in environment")

# Chat model choice (gpt-4o-mini by default)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# Embedding model for router & vector store
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = os.getenv("DATA_DIR", "data")
SESSION_DIR = os.getenv("SESSION_DIR", "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)  # ensure data dir exists

# Safety: limit retriever results and session length
MAX_FILES_TO_ROUTE = int(os.getenv("MAX_FILES_TO_ROUTE", "3"))
MAX_SNIPPETS = int(os.getenv("MAX_SNIPPETS", "8"))
MAX_SESSION_ENTRIES = int(os.getenv("MAX_SESSION_ENTRIES", "200"))

# -----------------------
# Clients / models
# -----------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embeddings_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Modern School Chatbot", version="1.0")

# Static mounts used by your frontend index.html
if os.path.isdir("img"):
    app.mount("/img", StaticFiles(directory="img"), name="img")
if os.path.isdir("css"):
    app.mount("/css", StaticFiles(directory="css"), name="css")
if os.path.isdir("dist"):
    app.mount("/dist", StaticFiles(directory="dist"), name="dist")

@app.get("/")
def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"status": "ok", "message": "Modern School Chatbot API running"})

# CORS is expected to be handled by Cloud Run / frontend. If you need middleware,
# you can add CORSMiddleware externally in deployment or uncomment below.
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -----------------------
# Dataset descriptions for router (meta-index)
# Keep this aligned with files in /data
# -----------------------
FILE_DESCRIPTIONS = {
    "about_school.txt": "general info, history, mission, and overview of Modern Senior Secondary School",
    "address.txt": "address, contact details, phone numbers and location of the school",
    "admission.txt": "admission procedures, eligibility, application process and dates",
    "staff.txt": "principal, vice-principal, headmistress, teachers and staff contact roles",
    "fees.txt": "fee structure, term fees, annual charges and payment instructions",
    "holidays.txt": "list of holidays and vacation schedules",
    "curriculum.txt": "subjects offered and high-level curriculum details",
    "extra_curricular.txt": "clubs, sports, arts and extracurricular activities",
    "infrastructure.txt": "labs, library, auditorium, playground and facilities",
    "competition_awards.txt": "awards and competition achievements",
    "ncert_math.txt": "NCERT math topics and formulas for school curriculum",
    "ncert_physics.txt": "NCERT physics formulas and concepts for school",
    "ncert_chemistry.txt": "NCERT chemistry topics and essential definitions",
    "social_studies.txt": "history, geography and civics overview",
    "school_toppers.txt": "records of high performing students and toppers",
    "student.txt": "student-related information and activities",
    "common_queries.txt": "commonly asked questions: timings, uniform, meetings, transports"
}

# If you have additional data files in data/ that are not listed, router will still use them by filename fallback.

# -----------------------
# Caches and router state
# -----------------------
# VECTOR_CACHE maps absolute file path -> LangChain retriever
VECTOR_CACHE: Dict[str, any] = {}

# Router descriptor embeddings: numpy matrix shape (n_files, dim)
ROUTER_EMBS: Optional[np.ndarray] = None
ROUTER_ORDER: List[str] = []  # list of filenames in same order as ROUTER_EMBS rows

# -----------------------
# Helper utilities
# -----------------------
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text_to_documents(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " "])
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]

def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

# -----------------------
# Build router embeddings from FILE_DESCRIPTIONS and data directory
# -----------------------
def build_router_index() -> None:
    global ROUTER_EMBS, ROUTER_ORDER
    # Use filenames present in data directory and FILE_DESCRIPTIONS keys
    files_in_data = sorted([os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.txt"))])
    # prefer files_in_data ordering, but include any known descriptions
    ordered = []
    for name in files_in_data:
        ordered.append(name)
    # ensure any described files present in ordered
    for fname in FILE_DESCRIPTIONS.keys():
        if fname not in ordered:
            # if file missing but described, still include description row (use fname)
            ordered.append(fname)
    ROUTER_ORDER = ordered
    descriptions = []
    for fname in ROUTER_ORDER:
        desc = FILE_DESCRIPTIONS.get(fname)
        if not desc:
            # fallback: use first 200 chars of file if exists, else file name
            p = os.path.join(DATA_DIR, fname)
            if os.path.exists(p):
                try:
                    text = read_text_file(p)
                    desc = (text[:200] + "...") if len(text) > 200 else text
                except Exception:
                    desc = fname
            else:
                desc = fname
        descriptions.append(desc)
    # embed all descriptions
    emb_list = []
    for d in descriptions:
        try:
            emb = embeddings_model.embed_query(d)
        except Exception as e:
            log.warning("Router embedding failed for description: %s", e)
            emb = [0.0] * 1536
        emb_list.append(emb)
    ROUTER_EMBS = l2_normalize_rows(np.array(emb_list, dtype=float))
    log.info("Router index built with %d entries", len(ROUTER_ORDER))

# run router build at import
try:
    build_router_index()
except Exception as e:
    log.warning("Failed to build router index at startup: %s", e)
    ROUTER_EMBS = None
    ROUTER_ORDER = []

# -----------------------
# Load/create FAISS retriever per file (cached)
# -----------------------
def load_vector_retriever_for_file(filename: str):
    path = os.path.join(DATA_DIR, filename)
    abs_path = os.path.abspath(path)
    if abs_path in VECTOR_CACHE:
        return VECTOR_CACHE[abs_path]
    if not os.path.exists(abs_path):
        log.warning("Data file missing for vector retrieval: %s", abs_path)
        return None
    try:
        text = read_text_file(abs_path)
        docs = split_text_to_documents(text)
        vs = FAISS.from_documents(docs, embeddings_model)
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        VECTOR_CACHE[abs_path] = retriever
        log.info("Built FAISS retriever for %s (%d chunks)", filename, len(docs))
        return retriever
    except Exception as e:
        log.warning("Failed to build FAISS retriever for %s: %s", filename, e)
        return None

# -----------------------
# Semantic routing + keyword fallback
# -----------------------
# Small keyword map for common fast matches
KEYWORD_MAP = {
    "fee": "fees.txt", "fees": "fees.txt", "tuition": "fees.txt",
    "principal": "staff.txt", "teacher": "staff.txt", "staff": "staff.txt",
    "address": "address.txt", "where": "address.txt",
    "admission": "admission.txt", "apply": "admission.txt",
    "holiday": "holidays.txt", "vacation": "holidays.txt", "holidays": "holidays.txt",
    "infrastructure": "infrastructure.txt", "facility": "infrastructure.txt",
    "toppers": "school_toppers.txt", "topper": "school_toppers.txt",
    "math": "ncert_math.txt", "physics": "ncert_physics.txt", "chemistry": "ncert_chemistry.txt",
    "curriculum": "curriculum.txt", "activity": "extra_curricular.txt", "club": "extra_curricular.txt",
    "faq": "common_queries.txt", "common": "common_queries.txt",
}

def route_files_for_query(query: str, max_files: int = MAX_FILES_TO_ROUTE) -> List[str]:
    ql = query.lower()
    matched = []
    # keyword quick matches
    for kw, fname in KEYWORD_MAP.items():
        if kw in ql and fname not in matched:
            matched.append(fname)
            if len(matched) >= max_files:
                break
    # semantic routing for additional files if needed
    if len(matched) < max_files and ROUTER_EMBS is not None and len(ROUTER_ORDER) > 0:
        try:
            q_emb = embeddings_model.embed_query(query)
            q_vec = np.array(q_emb, dtype=float)
            if np.linalg.norm(q_vec) == 0:
                q_vec = q_vec + 1e-10
            q_vec = q_vec / (np.linalg.norm(q_vec) or 1.0)
            sims = ROUTER_EMBS @ q_vec
            top_idx = list(np.argsort(-sims)[:max_files])
            for i in top_idx:
                fname = ROUTER_ORDER[i]
                if fname not in matched:
                    matched.append(fname)
                if len(matched) >= max_files:
                    break
        except Exception as e:
            log.warning("Semantic routing error: %s", e)
    # final fallback: use files actually present in /data
    present_files = [os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.txt"))]
    final = [f for f in matched if f in present_files]
    if not final:
        # if none matched, choose top present files from router order
        for fname in ROUTER_ORDER:
            if fname in present_files:
                final.append(fname)
            if len(final) >= max_files:
                break
    return final[:max_files]

# -----------------------
# Session management (per-user JSON files)
# -----------------------
def session_filename_for_user(user_id: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (user_id or "default_user"))
    return os.path.join(SESSION_DIR, f"{safe}.json")

def load_session_history(user_id: str) -> List[Dict]:
    path = session_filename_for_user(user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_session_history(user_id: str, history: List[Dict]) -> None:
    path = session_filename_for_user(user_id)
    try:
        # keep only the most recent MAX_SESSION_ENTRIES for disk
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history[-MAX_SESSION_ENTRIES:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Failed to save session for %s: %s", user_id, e)

# -----------------------
# Formatting helper
# -----------------------
def clean_answer_spacing(text: str) -> str:
    if not text:
        return ""
    # split on double newlines and single line breaks, keep paragraphs
    lines = [ln.strip() for ln in text.splitlines()]
    paras = []
    cur = []
    for ln in lines:
        if ln == "":
            if cur:
                paras.append(" ".join(cur).strip())
                cur = []
        else:
            cur.append(ln)
    if cur:
        paras.append(" ".join(cur).strip())
    # ensure blank line between paragraphs
    return "\n\n".join(p for p in paras if p)

# -----------------------
# Core retrieval + generation
# -----------------------
def get_answer_for_query(query: str, user_id: str) -> Dict:
    # 1) choose relevant files
    chosen_files = route_files_for_query(query, max_files=MAX_FILES_TO_ROUTE)
    context_snippets = []
    # 2) fetch relevant docs from each chosen file
    for fname in chosen_files:
        retriever = load_vector_retriever_for_file(fname)
        if not retriever:
            continue
        try:
            docs = retriever.get_relevant_documents(query)
            for d in docs:
                snippet = d.page_content.strip()
                if snippet:
                    header = f"--- source: {fname} ---"
                    context_snippets.append(f"{header}\n{snippet}")
        except Exception as e:
            log.warning("Error retrieving documents from %s: %s", fname, e)
    context_text = "\n\n".join(context_snippets[:MAX_SNIPPETS]) if context_snippets else ""

    # 3) construct prompt with instructions
    system_instructions = (
        "You are Brightly, the official assistant for Modern Senior Secondary School, Chennai. "
        "Use the context provided to answer accurately. "
        "If the answer is not found in context, say you don't have that information and suggest where to check. "
        "Be concise. Use blank lines between paragraphs. Avoid opinionated content."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    # 4) call OpenAI chat
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        # extract message text robustly
        raw = ""
        try:
            raw = response.choices[0].message.content
        except Exception:
            try:
                raw = response.choices[0].text
            except Exception:
                raw = ""
        answer = clean_answer_spacing(raw or "").strip()
    except Exception as e:
        log.warning("OpenAI generation error: %s", e)
        answer = "I am having trouble accessing the model right now. Please try again later."

    # 5) update user session history
    history = load_session_history(user_id)
    entry = {"question": query, "answer": answer, "sources": chosen_files, "ts": datetime.utcnow().isoformat()}
    history.append(entry)
    save_session_history(user_id, history)

    return {"answer": answer, "history": history, "sources": chosen_files}

# -----------------------
# Endpoints
# -----------------------
@app.post("/chat")
async def chat_endpoint(req: Request):
    try:
        payload = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON payload"}, status_code=400)
    question = (payload.get("question") or payload.get("message") or "").strip()
    user_id = payload.get("user_id", "default_user")
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    result = get_answer_for_query(question, user_id)
    return JSONResponse(result)

@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "service": "Modern School Chatbot"})

# -----------------------
# Local run
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
