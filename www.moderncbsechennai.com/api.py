# api.py
import os
import json
import math
import re
import random
import logging
import hashlib
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Try to import vector loader from your repo (must exist)
try:
    from vector import load_vector_store
except Exception:
    load_vector_store = None

# Try to import OpenAI new client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------
# Logging / debug
# ----------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("msss")

# ----------------------
# Config / env
# ----------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "modernschoolnanganallurchatbot")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
REFRESH_VECTORS_ON_STARTUP = os.getenv("REFRESH_VECTORS_ON_STARTUP", "true").lower() == "true"
NETLIFY_ORIGIN = os.getenv("NETLIFY_ORIGIN", "https://modernschoolnanganallur.netlify.app").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ----------------------
# OpenAI client wrappers
# ----------------------
_openai_client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        log.error("OpenAI client init failed: %s", e)
        _openai_client = None
else:
    if OpenAI is None:
        log.warning("openai package not installed; embedding/llm calls will fallback.")
    elif not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set; embedding/llm calls will fallback.")

class FallbackEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
    def embed_query(self, text: str):
        if not text:
            return [0.0] * self.dim
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        prev = h
        while len(vec) < self.dim:
            prev = hashlib.sha256(prev).digest()
            vec.extend([b / 255.0 for b in prev])
        return vec[:self.dim]

class OpenAIEmbedder:
    def __init__(self, client, model=OPENAI_EMBEDDING_MODEL, dim: int = 1536):
        self.client = client
        self.model = model
        self.dim = dim
    def embed_query(self, text: str):
        if not self.client:
            return FallbackEmbedder(self.dim).embed_query(text)
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

class OpenAIChat:
    def __init__(self, client, model=OPENAI_CHAT_MODEL, system_prompt: Optional[str]=None):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or ("You are Brightly, the official AI assistant for Modern Senior Secondary School, Chennai. "
                                               "Answer concisely in teacher-style English; prefer short steps for math solutions.")
    def invoke(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024):
        if not self.client:
            raise RuntimeError("OpenAI client not available")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # robust extraction
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg and getattr(msg, "content", None):
            return msg.content.strip()
        return (getattr(choice, "text", "") or getattr(resp, "output_text", "") or "").strip()

# Lazy singletons
_embedding_model = None
_answer_llm = None
_emotion_llm = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if _openai_client:
            _embedding_model = OpenAIEmbedder(_openai_client)
        else:
            _embedding_model = FallbackEmbedder(dim=512)
    return _embedding_model

def get_answer_llm():
    global _answer_llm
    if _answer_llm is None:
        if _openai_client:
            _answer_llm = OpenAIChat(_openai_client)
        else:
            _answer_llm = None
    return _answer_llm

def get_emotion_llm():
    global _emotion_llm
    if _emotion_llm is None:
        if _openai_client:
            _emotion_llm = OpenAIChat(_openai_client)
        else:
            _emotion_llm = None
    return _emotion_llm

# ----------------------
# App + CORS + Static
# ----------------------
app = FastAPI(title="MSSS Backend", version="1.0")

allowed_origins = [NETLIFY_ORIGIN] if NETLIFY_ORIGIN else []
if os.getenv("ALLOW_LOCALHOST", "true").lower() == "true":
    allowed_origins += ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"]
extra = [o.strip() for o in os.getenv("EXTRA_CORS_ORIGINS", "").split(",") if o.strip()]
allowed_origins += extra
if not allowed_origins:
    allowed_origins = ["*"]

log.info("CORS allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for d in ("img", "css", "dist"):
    if os.path.isdir(d):
        app.mount(f"/{d}", StaticFiles(directory=d), name=d)

# ----------------------
# State / persistence
# ----------------------
conversation_history: List[dict] = []
session_memory: List[dict] = []
vector_stores = {}

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("sessions", exist_ok=True)

# ----------------------
# Helpers (math, memory, vector)
# ----------------------
def solve_math_expression(expr: str):
    try:
        import sympy as sp
        expr_clean = expr.lower().replace("^", "**").replace("×", "*").replace("÷", "/").strip()
        x = sp.symbols("x")
        # simple equations
        if "=" in expr_clean:
            lhs, rhs = expr_clean.split("=", 1)
            sol = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            if not sol:
                return "No real solution found."
            if len(sol) == 1:
                return f"The value of x is {sol[0]}"
            return f"Possible values of x are: {', '.join(map(str, sol))}"
        # numeric eval fallback (guarded)
        allowed = {"sqrt": math.sqrt, "log": math.log10, "ln": math.log, "pi": math.pi, "e": math.e, "pow": pow}
        result = eval(expr_clean, {"__builtins__": None}, allowed)
        return f"The result is {round(result,6) if isinstance(result, float) else result}"
    except Exception as e:
        log.warning("Math eval error: %s", e)
        return None

def explain_math_step_by_step(expr: str):
    try:
        import sympy as sp
        x = sp.symbols("x")
        s = expr.lower().replace("^", "**").replace("×", "*")
        if "differentiate" in s or "derivative" in s:
            target = s.split("of")[-1].strip()
            func = sp.sympify(target)
            return f"The derivative of {func} wrt x is {sp.diff(func, x)}"
        if "integrate" in s:
            target = s.split("of")[-1].strip()
            func = sp.sympify(target)
            return f"The integral is {sp.integrate(func, x)} + C"
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            sol = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            return f"Solved: {sol}"
        return f"Simplified: {sp.simplify(s)}"
    except Exception:
        return None

def add_to_memory(question: str, answer: str):
    try:
        embed = get_embedding_model().embed_query(question)
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        embed = None
    session_memory.append({"question": question, "answer": answer, "embedding": embed})

def retrieve_relevant_memory(question: str, top_n: int = 5):
    try:
        q_emb = get_embedding_model().embed_query(question)
    except Exception as e:
        log.warning("Embedding error: %s", e)
        return ""
    if not session_memory:
        return ""
    def cosine(a,b):
        if not a or not b:
            return 0
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na==0 or nb==0:
            return 0
        return dot/(na*nb)
    scored = []
    for e in session_memory:
        score = cosine(q_emb, e.get("embedding"))
        scored.append((score, e))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [f"Q: {it['question']}\nA: {it['answer']}" for _, it in scored[:top_n]]
    return "\n".join(top)

def safe_retrieve(retriever, query: str):
    if retriever is None:
        return []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "similarity_search"):
            return retriever.similarity_search(query)
        return []
    except Exception as e:
        log.warning("Retriever error: %s", e)
        return []

# ----------------------
# Greetings / intents
# ----------------------
GREETINGS = ["hello","hi","hey","good morning","good afternoon","good evening"]
FAREWELLS = ["bye","goodbye","see you","farewell"]
INTENT_MAP = {
    "fees": ["fee","fees","structure","tuition"],
    "staff": ["principal","teacher","staff"],
    "address": ["address","location","contact"],
    "self_identity": ["who are you","your name","what are you","who created you"],
}

def check_greeting(q: str):
    ql = q.lower()
    if any(g in ql for g in GREETINGS):
        return "Welcome to Modern Senior Secondary School! I'm Brightly, your assistant. How can I help you today?"
    return None

def check_farewell(q: str):
    ql = q.lower()
    if any(f in ql for f in FAREWELLS):
        return "Goodbye! Have a great day 🌟 Come back soon!"
    return None

def detect_emotion(user_input: str):
    factual = ["what","where","when","how","who","which","fee","fees","address","location","principal","teacher","school","exam","contact","number","subject","student","class","admission"]
    if any(re.search(rf"\b{kw}\b", user_input.lower()) for kw in factual):
        return None
    if any(e in user_input for e in ["💡","😊","😄","🎉","🥳"]):
        return None
    prompt = f"Detect if this message is Positive / Negative / Neutral. Return one word.\nMessage: {user_input}"
    try:
        resp = get_emotion_llm().invoke(prompt).strip().capitalize() if get_emotion_llm() else None
        if resp == "Positive":
            return random.choice([
                "That's really kind of you, thank you 😊",
                "Glad to hear that! You're awesome!",
                "That made my day 😄",
            ])
        if resp == "Negative":
            return "I'm sorry if something felt off. Let's fix it."
    except Exception as e:
        log.warning("Emotion detect error: %s", e)
    return None

# ----------------------
# Vector store helpers
# ----------------------
def refresh_vector_stores():
    global vector_stores
    vector_stores = {}
    if load_vector_store is None:
        log.info("No vector.load_vector_store available; skipping vector build.")
        return
    data_dir = "data"
    if not os.path.isdir(data_dir):
        log.info("No data directory present; skipping vector build.")
        return
    for f in os.listdir(data_dir):
        if f.endswith(".txt"):
            path = os.path.join(data_dir, f)
            try:
                retr = load_vector_store(path)
                if retr:
                    vector_stores[os.path.splitext(f)[0]] = retr
            except Exception as e:
                log.warning("Failed to load vector %s: %s", path, e)
    log.info("Vector stores loaded: %s", list(vector_stores.keys()))

def load_ncert_vectors():
    # optional: attempts to load specific files if present
    datasets = {
        "ncert_maths": "data/ncert_maths.txt",
        "ncert_physics": "data/ncert_physics.txt",
        "ncert_chemistry": "data/ncert_chemistry.txt",
    }
    for name, path in datasets.items():
        if os.path.exists(path) and load_vector_store:
            try:
                retr = load_vector_store(path)
                if retr:
                    vector_stores[name] = retr
            except Exception as e:
                log.warning("Failed to load %s: %s", name, e)

# ----------------------
# Schemas
# ----------------------
class Query(BaseModel):
    question: str

# ----------------------
# Routes: health / root
# ----------------------
@app.get("/")
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({
        "message": "Modern Senior Secondary School — API online",
        "project": PROJECT_ID,
        "location": LOCATION,
        "vectors_refreshed_on_startup": REFRESH_VECTORS_ON_STARTUP
    })

@app.get("/health")
@app.get("/_/health")
def health():
    return {"status": "ok"}

@app.get("/llm/health")
def llm_health():
    try:
        llm = get_answer_llm()
        if llm is None:
            return JSONResponse(status_code=503, content={"ok": False, "error": "LLM not configured"})
        txt = llm.invoke("Say: Ok!").strip()
        return {"ok": bool(txt), "model": OPENAI_CHAT_MODEL}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.post("/chat")
async def chat(payload: dict):
    # legacy echo endpoint used earlier
    msg = payload.get("message", "")
    return {"reply": f"Echo: {msg}"}

@app.post("/ask")
async def ask(query: Query):
    q_text = query.question.strip()
    final_answers = []

    math_regex = re.compile(r"d/dx|dx|differentiate|derive|integrate|roots|equation|simplify|sin|cos|tan|log|sqrt|=|[\d+\-*/^()]")

    # 1) math direct
    if math_regex.search(q_text):
        step = explain_math_step_by_step(q_text)
        if step:
            add_to_memory(q_text, step)
            conversation_history.append({"question": q_text, "answer": step})
            return JSONResponse({"answer": step, "history": conversation_history})
        mr = solve_math_expression(q_text)
        if mr:
            add_to_memory(q_text, mr)
            conversation_history.append({"question": q_text, "answer": mr})
            return JSONResponse({"answer": mr, "history": conversation_history})

    # 2) greeting / farewell / emotion
    if resp := check_greeting(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := check_farewell(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := detect_emotion(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})

    # 3) identity / capabilities
    answer = None
    lq = q_text.lower()
    if any(phrase in lq for phrase in INTENT_MAP["self_identity"]):
        answer = "I'm Brightly — your friendly Modern Senior Secondary School assistant."
    elif any(word in lq for word in ["provide", "offer", "help", "assist", "what can you"]):
        answer = random.choice([
            "I can help you with school details, fees, admissions, exams, and staff information.",
            "I assist with queries about Modern Senior Secondary School — like fees, staff, or classes.",
            "I provide details about school activities, admissions, and academic info.",
            "I’m here to share school-related information and help you find what you need!"
        ])
    if answer:
        add_to_memory(q_text, answer)
        conversation_history.append({"question": q_text, "answer": answer})
        return JSONResponse({"answer": answer, "history": conversation_history})

    # 4) main retrieval + LLM
    sub_qs = re.split(r"[?;]| and ", q_text)
    sub_qs = [s.strip() for s in sub_qs if s.strip()]

    for sq in sub_qs:
        # build context from manual cache, vector stores, and session memory
        context = ""
        # manual cache
        MANUAL_CACHE_FILE = "answer_cache.json"
        if os.path.exists(MANUAL_CACHE_FILE):
            try:
                with open(MANUAL_CACHE_FILE, "r", encoding="utf-8") as f:
                    manual_cache = json.load(f)
                if sq in manual_cache:
                    context += str(manual_cache[sq].get("answer", "")) + "\n"
            except Exception as e:
                log.warning("Manual cache load error: %s", e)
        # vector stores
        for name, retr in vector_stores.items():
            try:
                docs = safe_retrieve(retr, sq)
                if docs:
                    context += "\n".join(getattr(d, "page_content", str(d)) for d in docs) + "\n"
            except Exception as e:
                log.warning("Retriever error (%s): %s", name, e)
        # session memory
        conv_ctx = retrieve_relevant_memory(sq)
        if conv_ctx:
            context += "\n--- Previous conversation ---\n" + conv_ctx
        if not context.strip():
            context = "No data found."

        prompt = f"""
You are Brightly — the official AI assistant for Modern Senior Secondary School, Chennai.
Explain and solve problems succinctly. Use teacher-style English and short steps for math.
Context:
{context}
Question: {sq}
Answer:
"""
        try:
            llm = get_answer_llm()
            if llm is None:
                raise RuntimeError("LLM not configured")
            answer = llm.invoke(prompt).strip()
        except Exception as e:
            log.warning("LLM error: %s", e)
            answer = "I’m having trouble accessing the model or data; please try again later."

        add_to_memory(sq, answer)
        conversation_history.append({"question": sq, "answer": answer})
        final_answers.append(answer)

    return JSONResponse({"answer": "\n".join(final_answers), "history": conversation_history})

# ----------------------
# Sessions persistence
# ----------------------
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE: Optional[str] = None

def start_new_session():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SESSION_DIR, f"session_{ts}.json")

def save_session_data(session_file: str):
    try:
        data = [{"question": q["question"], "answer": q["answer"]} for q in session_memory[-50:]]
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Failed to save session: %s", e)

def cleanup_old_sessions(max_files: int = 10):
    try:
        files = sorted([os.path.join(SESSION_DIR,f) for f in os.listdir(SESSION_DIR) if f.endswith(".json")], key=os.path.getmtime)
        for f in files[:-max_files]:
            try:
                os.remove(f)
                log.info("Deleted old session: %s", f)
            except Exception:
                pass
    except Exception as e:
        log.warning("Session cleanup error: %s", e)

# ----------------------
# Startup/shutdown
# ----------------------
@app.on_event("startup")
def startup_event():
    log.info("Server starting up. Project=%s, Location=%s", PROJECT_ID, LOCATION)
    cleanup_old_sessions()
    if REFRESH_VECTORS_ON_STARTUP:
        refresh_vector_stores()
        load_ncert_vectors()
        log.info("Vector stores loaded: %s", list(vector_stores.keys()))
    else:
        log.info("Skipping vector refresh on startup.")

@app.on_event("shutdown")
def shutdown_event():
    log.info("Server shutting down.")
    global SESSION_FILE
    if SESSION_FILE is None:
        SESSION_FILE = start_new_session()
    save_session_data(SESSION_FILE)
    cleanup_old_sessions()
    log.info("Session data saved to %s", SESSION_FILE)

# ----------------------
# Local dev entrypoint
# ----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
