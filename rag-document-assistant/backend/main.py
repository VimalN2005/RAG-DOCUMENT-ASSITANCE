"""
FastAPI server — Document upload, chat (RAG + LLM), and stats endpoints.
"""

import os
import shutil
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import groq  # pip install groq  (free, fast — swap for openai if preferred)

from rag_pipeline import RAGPipeline, UPLOAD_DIR

load_dotenv()

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(title="RAG Document Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    model: Optional[str] = "llama3-8b-8192"


class ChatResponse(BaseModel):
    answer: str
    sources: list
    model: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG Document Assistant is running 🚀"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF / TXT / DOCX)."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = rag.ingest(save_path, source_name=file.filename)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

    return {"filename": file.filename, **result}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question — retrieves relevant chunks, then calls LLM."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    chunks = rag.retrieve(req.question, top_k=req.top_k)

    if not chunks:
        return ChatResponse(
            answer="No documents are indexed yet. Please upload a document first.",
            sources=[],
            model=req.model,
        )

    # Build context
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
def stats():
    """Return index statistics."""
    return rag.stats()


@app.delete("/reset")
def reset_index():
    """Clear the entire vector index (use with caution)."""
    import data.faiss as _  # noqa
    rag.index.reset()
    rag.metadata.clear()
    rag._save()
    return {"status": "index cleared"}


# ──────────────────────────────────────────────
# Dev run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
