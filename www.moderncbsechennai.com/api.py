# api.py
import os
import json
import math
import re
import random
import logging
import hashlib
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# OpenAI client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Attempt to import your vector loader; fallback to None
try:
    from vector import load_vector_store
except Exception as e:
    load_vector_store = None
    # We will log after logger is initialized

# ======================
# Logging / Debug
# ======================
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("msss")

# ----------------------
# Defaults for your deployment (can be overridden by env)
# ----------------------
DEFAULT_PROJECT = "modernschoolnanganallurChatbot"
DEFAULT_LOCATION = "asia-south1"

def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

GOOGLE_CLOUD_PROJECT = env("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
GOOGLE_CLOUD_LOCATION = env("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
REFRESH_VECTORS_ON_STARTUP = env("REFRESH_VECTORS_ON_STARTUP", "true").lower() == "true"

# ======================
# OpenAI client setup
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OpenAI is None:
    log.error("OpenAI python package not available. Install `openai`.")
    _openai_client = None
else:
    try:
        # new OpenAI client usage
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        _openai_client = None
        log.error(f"Failed to initialize OpenAI client: {e}")

# ======================
# Embedding & LLM wrappers (replace Vertex)
# ======================
class OpenAIEmbedder:
    def __init__(self, client: "OpenAI", model: str = "text-embedding-3-small", dim: int = 512):
        self.client = client
        self.model = model
        self.dim = dim

    def embed_query(self, text: str):
        if not text:
            return [0.0] * self.dim
        if self.client is None:
            # fallback deterministic hashing to vector
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = []
            prev = h
            while len(vec) < self.dim:
                prev = hashlib.sha256(prev).digest()
                vec.extend([b / 255.0 for b in prev])
            return vec[:self.dim]
        try:
            # call OpenAI embeddings API
            res = self.client.embeddings.create(model=self.model, input=text)
            emb = res.data[0].embedding
            return emb
        except Exception as e:
            log.warning(f"Embedding call failed: {e}; using fallback embedder.")
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = []
            prev = h
            while len(vec) < self.dim:
                prev = hashlib.sha256(prev).digest()
                vec.extend([b / 255.0 for b in prev])
            return vec[:self.dim]

class OpenAIChatLLM:
    def __init__(self, client: "OpenAI", model: str = "gpt-4o-mini", system_prompt: str | None = None):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or (
            "You are Brightly, the official AI assistant of Modern Senior Secondary School, Chennai. "
            "Answer in a concise, helpful, teacher-style manner."
        )

    def invoke(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024):
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        try:
            # Use the Chat Completions API via client.chat.completions.create
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # New client returns choices with message content
            content = ""
            if resp and getattr(resp, "choices", None):
                # handle different shapes robustly
                choice = resp.choices[0]
                # some clients provide message or delta
                message = getattr(choice, "message", None)
                if message and getattr(message, "content", None):
                    content = message.content
                else:
                    # fallback to text or output_text
                    content = getattr(choice, "text", "") or getattr(resp, "output_text", "") or ""
            return content.strip()
        except Exception as e:
            raise

# Emotion detection uses same chat model but short prompt
class OpenAIEmotionLLM:
    def __init__(self, client: "OpenAI", model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def invoke(self, prompt: str):
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=32,
            )
            content = ""
            if resp and getattr(resp, "choices", None):
                choice = resp.choices[0]
                message = getattr(choice, "message", None)
                if message and getattr(message, "content", None):
                    content = message.content
                else:
                    content = getattr(choice, "text", "") or getattr(resp, "output_text", "") or ""
            return content.strip()
        except Exception as e:
            raise

# Lazy constructors for embedding and llms
_embedding_model = None
_answer_llm = None
_emotion_llm = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbedder(client=_openai_client, model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), dim=512)
    return _embedding_model

def get_answer_llm():
    global _answer_llm
    if _answer_llm is None:
        _answer_llm = OpenAIChatLLM(client=_openai_client, model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    return _answer_llm

def get_emotion_llm():
    global _emotion_llm
    if _emotion_llm is None:
        _emotion_llm = OpenAIEmotionLLM(client=_openai_client, model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    return _emotion_llm

# ----------------------
# Globals / state
# ----------------------
conversation_history = []
session_memory = []
vector_stores = {}

os.makedirs("vectorstore", exist_ok=True)
os.makedirs("sessions", exist_ok=True)
for _d in ("img", "css", "dist"):
    os.makedirs(_d, exist_ok=True)

# ======================
# FastAPI app + CORS
# ======================
app = FastAPI(title="MSSS Backend", version="1.0.0")

# ----------------------
# CORS setup (Netlify + Local)
# ----------------------
allowed_origins = ["https://modernschooltesting.netlify.app"]

log.info(f"🔐 CORS allow_origins = {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ----------------------
# Static mounts
# ----------------------
if os.path.isdir("img"):
    app.mount("/img", StaticFiles(directory="img"), name="img")
if os.path.isdir("css"):
    app.mount("/css", StaticFiles(directory="css"), name="css")
if os.path.isdir("dist"):
    app.mount("/dist", StaticFiles(directory="dist"), name="dist")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    if DEBUG:
        log.debug(f"➡️  {request.method} {request.url.path}")
    response = await call_next(request)
    if DEBUG:
        log.debug(f"⬅️  {request.method} {request.url.path} -> {response.status_code}")
    return response

@app.get("/")
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({
        "message": "Modern Senior Secondary School — API online",
        "project": GOOGLE_CLOUD_PROJECT,
        "location": GOOGLE_CLOUD_LOCATION,
        "vectors_refreshed_on_startup": REFRESH_VECTORS_ON_STARTUP
    })

@app.get("/health")
@app.get("/_/health")
def health():
    return {"status": "ok"}

@app.get("/llm/health")
def llm_health():
    try:
        txt = get_answer_llm().invoke("Say: Ok!").strip()
        return {"ok": bool(txt), "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), "text": txt or "(empty)"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.post("/chat")
async def chat(payload: dict):
    msg = payload.get("message", "")
    return {"reply": f"Echo: {msg}"}

# ======================
# Math utilities
# ======================
def solve_math_expression(expr: str):
    """
    Degree-based trig; simple equation solving with SymPy; safe eval sandbox.
    """
    try:
        import sympy as sp

        expr = (
            expr.lower()
            .replace("^", "**")
            .replace("×", "*")
            .replace("÷", "/")
            .strip()
        )

        x, y, z = sp.symbols("x y z")
        allowed = {
            "sin": lambda deg: math.sin(math.radians(float(deg))),
            "cos": lambda deg: math.cos(math.radians(float(deg))),
            "tan": lambda deg: math.tan(math.radians(float(deg))),
            "asin": lambda val: math.degrees(math.asin(float(val))),
            "acos": lambda val: math.degrees(math.acos(float(val))),
            "atan": lambda val: math.degrees(math.atan(float(val))),
            "sqrt": math.sqrt,
            "log": math.log10,
            "ln": math.log,
            "pi": math.pi,
            "e": math.e,
            "pow": pow,
        }

        if "=" in expr:
            lhs, rhs = expr.split("=")
            solution = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            if not solution:
                return "No real solution found."
            if len(solution) == 1:
                return f"The value of x is {solution[0]}."
            return f"Possible values of x are: {', '.join(map(str, solution))}."

        try:
            simplified = sp.simplify(expr)
            if str(simplified) != expr:
                expr = str(simplified)
        except Exception:
            pass

        result = eval(expr, {"__builtins__": None}, allowed)  # guarded
        if isinstance(result, float):
            result = round(result, 6)
        return f"The result is {result}"

    except Exception as e:
        log.warning(f"⚠️ Math solver error: {e}")
        return None


def explain_math_step_by_step(expr: str):
    import sympy as sp
    x, y, z = sp.symbols("x y z")
    try:
        expr = expr.lower().replace("^", "**").replace("×", "*")
        if ("differentiate" in expr) or ("derivative" in expr) or ("find dy/dx" in expr):
            target = expr.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.diff(func, x)
            return f"The derivative of {func} with respect to x is: {result}"
        elif ("integrate" in expr) or ("integration" in expr):
            target = expr.split("of")[-1].strip()
            func = sp.sympify(target)
            result = sp.integrate(func, x)
            return f"The integral of {func} with respect to x is: {result} + C"
        elif "=" in expr:
            lhs, rhs = expr.split("=")
            solution = sp.solve(sp.sympify(lhs) - sp.sympify(rhs), x)
            steps = [
                f"Step 1️⃣: Start with {lhs} = {rhs}",
                f"Step 2️⃣: Move all terms to one side: ({lhs}) - ({rhs}) = 0",
                f"Step 3️⃣: Simplify and solve for x",
                f"✅ Solution: x = {solution}",
            ]
            return "\n".join(steps)
        else:
            simplified = sp.simplify(expr)
            return f"Simplified form: {simplified}"
    except Exception:
        return None

# ======================
# Memory helpers
# ======================
def add_to_memory(question: str, answer: str):
    try:
        embed = get_embedding_model().embed_query(question)
    except Exception as e:
        log.warning(f"⚠️ Embedding error: {e}")
        embed = None
    session_memory.append({"question": question, "answer": answer, "embedding": embed})

def retrieve_relevant_memory(question: str, top_n=5):
    try:
        query_embed = get_embedding_model().embed_query(question)
    except Exception as e:
        log.warning(f"⚠️ Embedding error: {e}")
        return ""

    if not session_memory:
        return ""

    def cosine_similarity(a, b):
        if a is None or b is None:
            return 0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    scored = []
    for entry in session_memory:
        score = cosine_similarity(query_embed, entry["embedding"])
        scored.append((score, entry))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_entries = [f"Q: {e['question']}\nA: {e['answer']}" for _, e in scored[:top_n]]
    return "\n".join(top_entries)

# ======================
# Greetings / Farewells / Emotion
# ======================
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
FAREWELLS = ["bye", "goodbye", "see you", "farewell"]

def check_greeting(q: str):
    q = q.lower()
    if any(g in q for g in GREETINGS):
        return ("Welcome to Modern Senior Secondary School! I'm Brightly, your assistant. "
                "How can I help you today?")
    return None

def check_farewell(q: str):
    q = q.lower()
    if any(f in q for f in FAREWELLS):
        return "Goodbye! Have a great day 🌟 Come back soon!"
    return None

def detect_emotion(user_input: str):
    factual_keywords = [
        "what","where","when","how","who","which","fee","fees","address","location",
        "principal","teacher","school","exam","contact","number","subject","student",
        "class","admission",
    ]
    if any(re.search(rf"\b{kw}\b", user_input.lower()) for kw in factual_keywords):
        return None
    if any(emoji in user_input for emoji in ["💡", "😊", "😄", "🎉", "🥳"]):
        return None
    prompt = f"""
Detect if this message is Positive (appreciation/humor) or Negative (complaint/anger).
Return only: Positive / Negative / Neutral
Message: {user_input}
"""
    try:
        resp = get_emotion_llm().invoke(prompt).strip().capitalize()
        if resp == "Positive":
            responses = [
                "That's really kind of you, thank you 😊",
                "Glad to hear that! You're awesome!",
                "That made my day 😄",
                "You're too sweet — thanks a lot!",
                "Aww, I appreciate that 💫",
            ]
            return random.choice(responses)
        elif resp == "Negative":
            return "I'm sorry if something felt off. Let’s fix it together."
    except Exception as e:
        log.warning(f"⚠️ Emotion detection error: {e}")
    return None

# ======================
# Intent & Vector Stores
# ======================
INTENT_MAP = {
    "fees": ["fee", "fees", "structure", "tuition"],
    "staff": ["principal", "teacher", "staff"],
    "address": ["address", "location", "contact"],
    "self_identity": ["who are you", "your name", "what are you", "who created you"],
}

def refresh_vector_stores():
    """Build retrievers from all .txt files in ./data (if present)."""
    global vector_stores
    vector_stores = {}
    if load_vector_store is None:
        log.info("ℹ️ load_vector_store unavailable; skipping vector build.")
        return
    if not os.path.isdir("data"):
        log.info("ℹ️ No data directory found; skipping vector build.")
        return
    current_files = {
        os.path.splitext(f)[0]: os.path.join("data", f)
        for f in os.listdir("data")
        if f.endswith(".txt")
    }
    for name, file in current_files.items():
        try:
            retriever = load_vector_store() # Assuming load_vector_store no longer takes a path
            if retriever:
                vector_stores[name] = retriever
        except Exception as e:
            log.warning(f"⚠️ Failed to build retriever for '{file}': {e}")
    log.info(f"✅ Vector stores loaded: {list(vector_stores.keys())}")

# ======================
# Misc helpers
# ======================
def split_subquestions(q: str):
    if not any(sep in q.lower() for sep in [" and ", ";", "?"]):
        return [q.strip()]
    return [s.strip() for s in re.split(r"[?;]| and ", q) if s.strip()]

CLASS_MAP = {
    "lkg": "LKG", "ukg": "UKG", "1st": "I", "first": "I", "i": "I",
    "2nd": "II", "second": "II", "3rd": "III", "third": "III",
    "4th": "IV", "5th": "V", "6th": "VI", "7th": "VII", "8th": "VIII",
    "9th": "IX", "10th": "X", "tenth": "X",
    "11th cs": "XI-CS", "11th bio": "XI-BIO", "11th comm": "XI-COMM",
    "12th cs": "XII-CS", "12th bio": "XII-BIO", "12th comm": "XII-COMM",
}

def safe_retrieve(retriever, query):
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever._get_relevant_documents(query, run_manager=None)

# ======================
# NCERT optional datasets (if present)
# ======================
NCERT_DATASETS = {
    "ncert_maths": "data/ncert_maths.txt",
    "ncert_physics": "data/ncert_physics.txt",
    "ncert_chemistry": "data/ncert_chemistry.txt",
}

def load_ncert_vectors():
    if load_vector_store is None:
        log.info("ℹ️ load_vector_store unavailable; skipping NCERT.")
        return
    loaded = []
    for name, path in NCERT_DATASETS.items():
        if os.path.exists(path):
            try:
                retriever = load_vector_store() # Assuming load_vector_store no longer takes a path
                if retriever:
                    vector_stores[name] = retriever
                    loaded.append(name)
            except Exception as e:
                log.warning(f"⚠️ Failed to load NCERT '{name}': {e}")
    log.info(f"📘 NCERT datasets loaded: {loaded}")

# ======================
# Schemas
# ======================
class Query(BaseModel):
    question: str

# ======================
# Admin
# ======================
@app.post("/admin/refresh")
def admin_refresh():
    try:
        refresh_vector_stores()
        load_ncert_vectors()
        return {
            "ok": True,
            "message": "Vectors refreshed",
            "stores": list(vector_stores.keys()),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

# ======================
# Ask endpoint
# ======================

@app.post("/ask")
async def ask(query: Query):
    q_text = query.question.strip()
    final_answers = []

    math_regex = re.compile(
        r"d/dx|dx|differentiate|derive|integrate|roots|equation|simplify|sin|cos|tan|log|sqrt|=|[\d+\-*/^()]"
    )

    # 1) Math / Science direct
    if math_regex.search(q_text):
        step_result = explain_math_step_by_step(q_text)
        if step_result:
            add_to_memory(q_text, step_result)
            conversation_history.append({"question": q_text, "answer": step_result})
            return JSONResponse({"answer": step_result, "history": conversation_history})

        math_result = solve_math_expression(q_text)
        if math_result:
            add_to_memory(q_text, math_result)
            conversation_history.append({"question": q_text, "answer": math_result})
            return JSONResponse({"answer": math_result, "history": conversation_history})

    # 2) Greetings / Farewell / Emotion
    if resp := check_greeting(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := check_farewell(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})
    if resp := detect_emotion(q_text):
        return JSONResponse({"answer": resp, "history": conversation_history})

    # 3) Identity / capabilities
    answer = None
    lower_q = q_text.lower()
    if any(phrase in lower_q for phrase in INTENT_MAP["self_identity"]):
        answer = "I'm Brightly — your friendly Modern Senior Secondary School assistant."
    elif any(word in lower_q for word in ["provide", "offer", "help", "assist", "what can you"]):
        answer = random.choice(
            [
                "I can help you with school details, fees, admissions, exams, and staff information.",
                "I assist with queries about Modern Senior Secondary School — like fees, staff, or classes.",
                "I provide details about school activities, admissions, and academic info.",
                "I’m here to share school-related information and help you find what you need!",
            ]
        )

    if answer:
        add_to_memory(q_text, answer)
        conversation_history.append({"question": q_text, "answer": answer})
        return JSONResponse({"answer": answer, "history": conversation_history})

    # 4) Main processing (split)
    sub_qs = split_subquestions(q_text)

    simple_math_questions = {
        "quadratic equations": "A quadratic equation is of the form ax² + bx + c = 0. The solutions are x = [-b ± √(b² - 4ac)] / 2a.",
    }

    for sq in sub_qs:
        answer = None

        for key, val in simple_math_questions.items():
            if key in sq.lower():
                answer = val
                break

        if not answer and math_regex.search(sq):
            step_result = explain_math_step_by_step(sq)
            answer = step_result or solve_math_expression(sq)

        if not answer:
            context = ""
            # Manual cache (optional file)
            MANUAL_CACHE_FILE = "answer_cache.json"
            if os.path.exists(MANUAL_CACHE_FILE):
                try:
                    with open(MANUAL_CACHE_FILE, "r", encoding="utf-8") as f:
                        manual_cache = json.load(f)
                    if sq in manual_cache:
                        context += str(manual_cache[sq].get("answer", "")) + "\n"
                except Exception as e:
                    log.warning(f"⚠️ Manual cache load error: {e}")

            # Vector stores
            for store_name, retriever in vector_stores.items():
                try:
                    results = safe_retrieve(retriever, sq)
                    if results:
                        context += "\n".join([doc.page_content for doc in results]) + "\n"
                except Exception as e:
                    log.warning(f"⚠️ Retriever '{store_name}' error: {e}")

            # Conversation memory
            conv_context = retrieve_relevant_memory(sq)
            if conv_context:
                context += "\n--- Previous conversation ---\n" + conv_context
            if not context.strip():
                context = "No data found."

            prompt = f"""
You are called and are Brightly — the official AI assistant of Modern Senior Secondary School, Chennai.
You are also a helpful school AI tutor.
Explain and solve problems in math, physics, and chemistry like an experienced teacher.
When students ask a study question:
- Give short, clear explanations with steps.
- Use plain, teacher-style English.

Knowledge Scope:
- NCERT-based Physics, Chemistry, Maths (Classes 6–12)
- General school information

Tone:
- Friendly, conversational, natural
- Keep responses concise

Rules:
- Avoid politics; keep school-relevant
- If asked who created you, answer: "I was created by the technical team at Modern Senior Secondary School to assist students and parents with school-related queries."

Context:
{context}
Question: {sq}
Answer:
"""
            try:
                answer = get_answer_llm().invoke(prompt).strip()
            except Exception as e:
                log.warning(f"⚠️ LLM error: {e}")
                answer = "I’m having trouble accessing the data at the moment, please try again."

        add_to_memory(sq, answer)
        conversation_history.append({"question": sq, "answer": answer})
        final_answers.append(answer)

    return JSONResponse({"answer": "\n".join(final_answers), "history": conversation_history})

# ======================
# Sessions (optional persistence)
# ======================
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SESSION_FILE = None

def start_new_session():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SESSION_DIR, f"session_{timestamp}.json")

def save_session_data(session_file):
    try:
        data_to_save = [{"question": q["question"], "answer": q["answer"]} for q in session_memory[-50:]]
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"⚠️ Failed to save session: {e}")

def cleanup_old_sessions(max_files=10):
    try:
        files = sorted(
            [os.path.join(SESSION_DIR, f) for f in os.listdir(SESSION_DIR) if f.endswith(".json")],
            key=os.path.getmtime,
        )
        for f in files[:-max_files]:
            try:
                os.remove(f)
                log.info(f"🗑️ Deleted old session: {f}")
            except Exception:
                pass
    except Exception as e:
        log.warning(f"⚠️ Session cleanup error: {e}")

# ======================
# Startup / Shutdown
# ======================
@app.on_event("startup")
def startup_event():
    log.info("🚀 Server starting up...")
    log.info(f"   Project = {GOOGLE_CLOUD_PROJECT}")
    log.info(f"   Location = {GOOGLE_CLOUD_LOCATION}")
    log.info(f"   Refresh vectors on startup = {REFRESH_VECTORS_ON_STARTUP}")
    netlify_origin = os.getenv("NETLIFY_ORIGIN", "")
    if netlify_origin:
        log.info(f"NETLIFY_ORIGIN = {netlify_origin}")
    cleanup_old_sessions(max_files=10)

    if REFRESH_VECTORS_ON_STARTUP:
        refresh_vector_stores()
        load_ncert_vectors()
        log.info("✅ Vector stores + NCERT data loaded.")
    else:
        log.info("⏭️ Skipping vector refresh/load on startup.")

@app.on_event("shutdown")
def shutdown_event():
    log.info("🚪 Server shutting down...")
    global SESSION_FILE
    if SESSION_FILE is None:
        SESSION_FILE = start_new_session()
    save_session_data(SESSION_FILE)
    cleanup_old_sessions(max_files=10)
    log.info(f"💾 Session data saved to {SESSION_FILE}")

# Local dev entrypoint (Cloud Run uses your Docker CMD)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
