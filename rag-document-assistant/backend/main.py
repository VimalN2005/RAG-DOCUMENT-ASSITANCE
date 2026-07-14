"""
FastAPI server — Document upload, chat (RAG + LLM), and stats endpoints.
Session-based: each user gets an isolated, in-memory document index.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import groq  # pip install groq  (free, fast — swap for openai if preferred)

from rag_pipeline import session_manager

load_dotenv()

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(title="RAG Document Assistant", version="2.0.0")

# NOTE: allow_origins should be narrowed to your actual Vercel URL once deployed.
# Example: allow_origins=["https://your-frontend.vercel.app"]
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip() or "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET")  # required to hit /admin/* endpoints

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    model: Optional[str] = "llama-3.1-8b-instant"


class ChatResponse(BaseModel):
    answer: str
    sources: list
    model: str


# ──────────────────────────────────────────────
# Session helper
# ──────────────────────────────────────────────
def get_or_create_session(x_session_id: Optional[str] = Header(None)):
    """
    Resolve the caller's session.
    - If a valid, existing session id is provided in the X-Session-Id header, use it.
    - Otherwise, create a brand-new session and return its id along with the pipeline.
    """
    if x_session_id and session_manager.session_exists(x_session_id):
        session_id = x_session_id
    else:
        session_id = session_manager.create_session()

    pipeline = session_manager.get_pipeline(session_id)
    return session_id, pipeline


def require_admin(x_admin_secret: Optional[str] = Header(None)):
    if not ADMIN_SECRET or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Not authorized.")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG Document Assistant is running 🚀"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session: tuple = Depends(get_or_create_session),
):
    """Upload and index a document (PDF / TXT / DOCX) into the caller's session."""
    session_id, pipeline = session

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    # Use a temp file — nothing is written into a permanent/shared location.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest(tmp_path, source_name=file.filename)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        os.remove(tmp_path)

    return {"session_id": session_id, "filename": file.filename, **result}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: tuple = Depends(get_or_create_session),
):
    """Ask a question — retrieves relevant chunks from THIS session only, then calls LLM."""
    session_id, pipeline = session

    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    chunks = pipeline.retrieve(req.question, top_k=req.top_k)

    if not chunks:
        return ChatResponse(
            answer="No documents are indexed yet in this session. Please upload a document first.",
            sources=[],
            model=req.model,
        )

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk']}" for c in chunks
    )

    system_prompt = (
        "You are a precise document assistant. "
        "Answer questions ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't have enough information.' "
        "Be concise and cite the source document when possible."
    )

    user_prompt = f"""Context:
{context}

Question: {req.question}

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {str(e)}")

    sources = list({c["source"] for c in chunks})
    return ChatResponse(answer=answer, sources=sources, model=req.model)


@app.get("/stats")
def stats(session: tuple = Depends(get_or_create_session)):
    """Return index statistics for the caller's session only."""
    session_id, pipeline = session
    return {"session_id": session_id, **pipeline.stats()}


@app.delete("/reset")
def reset_session(session: tuple = Depends(get_or_create_session)):
    """Clear the caller's own session (start fresh) — does not affect other users."""
    session_id, _ = session
    session_manager.delete_session(session_id)
    new_session_id = session_manager.create_session()
    return {"status": "session cleared", "session_id": new_session_id}


# ──────────────────────────────────────────────
# Admin endpoints — require X-Admin-Secret header
# ──────────────────────────────────────────────

@app.get("/admin/sessions")
def admin_list_sessions(_: None = Depends(require_admin)):
    """List all active sessions (admin only)."""
    return {"active_sessions": session_manager.list_sessions()}


@app.delete("/admin/sessions/{session_id}")
def admin_kill_session(session_id: str, _: None = Depends(require_admin)):
    """Force-delete any session immediately (admin only)."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(404, "Session not found.")
    return {"status": "session deleted", "session_id": session_id}


# ──────────────────────────────────────────────
# Dev run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)