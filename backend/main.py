import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import fitz  # PyMuPDF
from docx import Document as DocxDocument

import numpy as np
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from dotenv import load_dotenv

# ------------------ Config & init ------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # if present

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = set(
    (os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt")).split(","))
SESSIONS_DIR = BASE_DIR / os.getenv("SESSIONS_DIR", "sessions")

SESSIONS_DIR.mkdir(exist_ok=True)

# Models
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini init
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini = None

# ------------------ FastAPI app ------------------
app = FastAPI(title="AI Knowledge Chatbot API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------------------ Session helpers ------------------


def get_session_paths(session_id: str):
    sdir = SESSIONS_DIR / session_id
    return {
        "root": sdir,
        "upload": sdir / "uploads",
        "index_dir": sdir / "faiss_index",
        "index_file": sdir / "faiss_index" / "index.faiss",
        "meta": sdir / f"document_metadata_{session_id}.json",
    }


def ensure_session_exists(session_id: str):
    paths = get_session_paths(session_id)
    if not paths["root"].exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return paths


def load_meta(session_id: str) -> dict:
    paths = get_session_paths(session_id)
    if paths["meta"].exists():
        with open(paths["meta"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "documents": {},
        "chunks": [],
        "title": "New Chat",
        "created_at": datetime.now().isoformat()
    }


def save_meta(session_id: str, meta: dict):
    paths = get_session_paths(session_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def generate_session_title(user_message: str) -> str:
    """Generate a ChatGPT-style title from the first user message"""
    if not gemini:
        # Fallback: simple truncation
        words = user_message.split()[:4]
        return " ".join(words) + ("..." if len(user_message.split()) > 4 else "")

    try:
        title_prompt = f"""Generate a short, descriptive title (3-5 words max) for a chat session based on this first message:

"{user_message}"

Rules:
- Be concise and descriptive
- No quotes or special characters
- Examples: "Python debugging help", "Travel planning advice", "Recipe recommendations"
- Just return the title, nothing else"""

        resp = gemini.generate_content(title_prompt)
        title = getattr(resp, "text", "").strip()

        # Fallback if generation fails
        if not title or len(title) > 50:
            words = user_message.split()[:4]
            title = " ".join(words) + \
                ("..." if len(user_message.split()) > 4 else "")

        return title
    except Exception:
        # Fallback on any error
        words = user_message.split()[:4]
        return " ".join(words) + ("..." if len(user_message.split()) > 4 else "")


def ext_ok(filename: str) -> bool:
    return filename.lower().split(".")[-1] in ALLOWED_EXTENSIONS

# ------------------ Extraction ------------------


def extract_pdf(path: Path) -> List[str]:
    doc = fitz.open(str(path))
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return pages


def extract_docx(path: Path) -> List[str]:
    doc = DocxDocument(str(path))
    return ["\n".join(p.text for p in doc.paragraphs if p.text.strip())]


def extract_txt(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [f.read()]

# ------------------ Vector utilities ------------------


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def write_index(index: Optional[faiss.IndexFlatIP], index_file: Path):
    if index is not None:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_file))


def read_index(dim: int, index_file: Path) -> Optional[faiss.IndexFlatIP]:
    if index_file.exists():
        idx = faiss.read_index(str(index_file))
        if idx.d != dim:
            return None
        return idx
    return None


def rebuild_index_from_chunks(session_id: str, meta: dict) -> Optional[faiss.IndexFlatIP]:
    texts = [c["text"] for c in meta.get("chunks", [])]
    paths = get_session_paths(session_id)
    index_file = paths["index_file"]

    if not texts:
        # clear index on disk if no chunks
        if index_file.exists():
            index_file.unlink()
        return None

    embs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    embs = normalize(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    write_index(index, index_file)
    return index


def load_or_build_index(session_id: str, meta: dict) -> Optional[faiss.IndexFlatIP]:
    texts = [c["text"] for c in meta.get("chunks", [])]
    if not texts:
        return None
    dim = embedder.get_sentence_embedding_dimension()
    paths = get_session_paths(session_id)
    idx = read_index(dim, paths["index_file"])
    if idx is None:
        idx = rebuild_index_from_chunks(session_id, meta)
    return idx


def top_k(session_id: str, meta: dict, query: str, k: int = 4) -> List[dict]:
    index = load_or_build_index(session_id, meta)
    if index is None or not meta.get("chunks"):
        return []
    emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    emb = normalize(emb)
    D, I = index.search(emb, k)
    out = []
    for rank, idx in enumerate(I[0].tolist()):
        if idx < 0 or idx >= len(meta["chunks"]):
            continue
        ch = meta["chunks"][idx]
        out.append(
            {"text": ch["text"], "metadata": ch["metadata"], "score": float(D[0][rank])})
    return out

# ------------------ Chunk registration ------------------


def chunk_and_register(session_id: str, filename: str, raw_pages: List[str]) -> bool:
    meta = load_meta(session_id)
    chunks_local = []
    for page_i, page_text in enumerate(raw_pages, start=1):
        for chunk_i, chunk in enumerate(text_splitter.split_text(page_text)):
            chunks_local.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": page_i,
                    "chunk_id": f"{filename}::p{page_i}::c{chunk_i}"
                }
            })
    if not chunks_local:
        return False

    meta["chunks"].extend(chunks_local)
    meta["documents"][filename] = {
        "filename": filename,
        "file_type": filename.split(".")[-1],
        "upload_date": datetime.now().isoformat(),
        "status": "processed",
        "page_count": len(raw_pages),
    }
    save_meta(session_id, meta)
    rebuild_index_from_chunks(session_id, meta)
    return True


def remove_document(session_id: str, filename: str):
    meta = load_meta(session_id)
    if filename in meta.get("documents", {}):
        del meta["documents"][filename]
    meta["chunks"] = [c for c in meta.get(
        "chunks", []) if c["metadata"]["source"] != filename]
    save_meta(session_id, meta)
    rebuild_index_from_chunks(session_id, meta)

# ------------------ Schemas ------------------


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    # [{"role": "user"|"assistant", "content": "..."}]
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_id: str
    session_title: Optional[str] = None  # New field for updated title

# ------------------ Session Routes ------------------


@app.post("/sessions")
def create_session():
    sid = str(uuid4())
    paths = get_session_paths(sid)
    paths["upload"].mkdir(parents=True, exist_ok=True)
    paths["index_dir"].mkdir(parents=True, exist_ok=True)

    # --- Short timestamp style ---
    now = datetime.now().strftime("%b%d, %H:%M")  # e.g., Aug18, 21:30
    title = f"Chat – {now}"

    save_meta(sid, {
        "documents": {},
        "chunks": [],
        "title": title,
        "created_at": datetime.now().isoformat()
    })
    return {"session_id": sid, "title": title}


@app.get("/sessions")
def list_sessions():
    if not SESSIONS_DIR.exists():
        return []

    sessions_info = []
    for p in SESSIONS_DIR.iterdir():
        if p.is_dir():
            meta = load_meta(p.name)
            sessions_info.append({
                "session_id": p.name,
                "title": meta.get("title", "New Chat"),
                "created_at": meta.get("created_at", "")
            })

    # Sort by creation date, newest first
    sessions_info.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions_info


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    paths = get_session_paths(session_id)
    if not paths["root"].exists():
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(paths["root"])
    return {"message": f"Session {session_id} deleted"}

# ------------------ Document Routes ------------------


@app.post("/upload")
async def upload(session_id: str = Query(...), file: UploadFile = File(...)):
    paths = ensure_session_exists(session_id)
    if not ext_ok(file.filename):
        raise HTTPException(
            status_code=400, detail=f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    size_mb = 0
    dest = paths["upload"] / file.filename
    with open(dest, "wb") as f:
        data = await file.read()
        f.write(data)
        size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413, detail=f"File too large (> {MAX_FILE_SIZE_MB} MB)")

    ext = file.filename.lower().split(".")[-1]
    if ext == "pdf":
        pages = extract_pdf(dest)
    elif ext == "docx":
        pages = extract_docx(dest)
    else:
        pages = extract_txt(dest)

    if not chunk_and_register(session_id, file.filename, pages):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail="No text extracted from document")

    return {"message": "uploaded", "filename": file.filename, "session_id": session_id}


@app.get("/documents")
def documents(session_id: str = Query(...)):
    ensure_session_exists(session_id)
    meta = load_meta(session_id)
    return list(meta.get("documents", {}).values())


@app.delete("/documents/{filename}")
def delete_document(filename: str, session_id: str = Query(...)):
    paths = ensure_session_exists(session_id)
    dest = paths["upload"] / filename
    if dest.exists():
        dest.unlink()
    remove_document(session_id, filename)
    return {"message": "deleted", "filename": filename, "session_id": session_id}

# ------------------ Chat Route ------------------


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session_id: str = Query(...)):
    if not gemini:
        raise HTTPException(
            status_code=500, detail="Gemini API key not configured")

    meta = load_meta(session_id)

    # Check if this is the first user message and generate title
    session_title = None
    is_first_message = meta.get("title") == "New Chat" and (
        not req.history or len(req.history) == 0)

    if is_first_message:
        new_title = generate_session_title(req.message)
        meta["title"] = new_title
        save_meta(session_id, meta)
        session_title = new_title

    hits = top_k(session_id, meta, req.message, k=4)
    if not hits:
        # Even without context, we should still return the new title if generated
        return ChatResponse(
            answer="I don't have relevant context yet for this chat. Please upload documents to this session and try again.",
            sources=[],
            conversation_id=req.conversation_id or session_id,
            session_title=session_title
        )

    context_blocks = []
    for h in hits:
        src = h["metadata"]["source"]
        page = h["metadata"]["page"]
        context_blocks.append(f"Source: {src} (page {page})\n{h['text']}")
    context = "\n\n-----\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If the answer is not contained there, say you cannot find it in the uploaded documents."
    )
    user_prompt = f"Question: {req.message}\n\nContext:\n{context}\n\nAnswer clearly and concisely."

    parts = []
    if req.history:
        for turn in req.history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append({"role": role, "parts": [content]})

    parts.append({"role": "user", "parts": [
                 system_prompt + "\n\n" + user_prompt]})
    resp = gemini.generate_content(parts)
    answer = getattr(resp, "text", "").strip() or "No response."

    sources = [{"filename": h["metadata"]["source"],
                "page": h["metadata"]["page"], "score": h["score"]} for h in hits]
    return ChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=req.conversation_id or session_id,
        session_title=session_title
    )


@app.get("/health")
def health():
    return {"status": "ok"}
