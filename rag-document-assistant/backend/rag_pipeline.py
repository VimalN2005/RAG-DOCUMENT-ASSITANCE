"""
RAG Pipeline — Core document ingestion, chunking, embedding, and retrieval.

Each user gets their own isolated, in-memory RAGPipeline instance (via
SessionManager). Nothing is persisted to disk — everything lives only for
the lifetime of the session.
"""

import time
import threading
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64

SESSION_INACTIVITY_TIMEOUT = 30 * 60      # 30 minutes since last activity
SESSION_MAX_LIFETIME       = 2 * 60 * 60  # 2 hours hard cap
CLEANUP_INTERVAL           = 5 * 60       # run cleanup every 5 minutes

# ──────────────────────────────────────────────
# Shared, stateless resources (loaded ONCE for all sessions)
# ──────────────────────────────────────────────
print("[RAG] Loading embedding model (shared across sessions) …")
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
_embed_dim = _embedder.get_sentence_embedding_dimension()

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)


def _load_file(filepath: str) -> List[str]:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath, encoding="utf-8")
    elif ext in (".doc", ".docx"):
        loader = Docx2txtLoader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    return [d.page_content for d in docs]


# ──────────────────────────────────────────────
# Per-session pipeline (in-memory only, no disk persistence)
# ──────────────────────────────────────────────
class RAGPipeline:
    """
    One instance per user session.
      1. Load & parse documents (PDF / TXT / DOCX)
      2. Chunk with overlap
      3. Embed with the shared sentence-transformer model
      4. Store in an in-memory FAISS flat-L2 index (this session only)
      5. Retrieve top-k chunks for a query (this session only)
    """

    def __init__(self):
        self.index = faiss.IndexFlatL2(_embed_dim)
        self.metadata: List[Dict] = []

    # ── Ingest ────────────────────────────────

    def ingest(self, filepath: str, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Parse, chunk, embed, and index a document (in memory only)."""
        source_name = source_name or Path(filepath).name
        file_hash = hashlib.md5(open(filepath, "rb").read()).hexdigest()

        existing = [m for m in self.metadata if m.get("hash") == file_hash]
        if existing:
            return {"status": "skipped", "reason": "already indexed", "source": source_name}

        raw_pages = _load_file(filepath)
        full_text = "\n\n".join(raw_pages)
        chunks = _splitter.split_text(full_text)

        if not chunks:
            return {"status": "error", "reason": "no content extracted"}

        vectors = _embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        vectors = vectors.astype("float32")

        start_idx = len(self.metadata)
        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id":     start_idx + i,
                "source": source_name,
                "hash":   file_hash,
                "chunk":  chunk,
            })

        return {"status": "ok", "chunks": len(chunks), "source": source_name}

    # ── Retrieve ──────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k most relevant chunks for a query (this session only)."""
        if self.index.ntotal == 0:
            return []

        q_vec = _embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "chunk":   meta["chunk"],
                "source":  meta["source"],
                "score":   float(dist),
            })
        return results

    # ── Stats ─────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        sources = list({m["source"] for m in self.metadata})
        return {
            "total_chunks": self.index.ntotal,
            "documents":    len(sources),
            "sources":      sources,
        }


# ──────────────────────────────────────────────
# Session manager — tracks one RAGPipeline per user
# ──────────────────────────────────────────────
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._sessions[session_id] = {
                "pipeline":    RAGPipeline(),
                "created_at":  now,
                "last_active": now,
            }
        return session_id

    def get_pipeline(self, session_id: Optional[str]) -> Optional[RAGPipeline]:
        """Return the pipeline for a session, refreshing its last_active time."""
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session["last_active"] = time.time()
            return session["pipeline"]

    def session_exists(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        with self._lock:
            return session_id in self._sessions

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            return [
                {
                    "session_id":        sid,
                    "created_at":        data["created_at"],
                    "last_active":       data["last_active"],
                    "idle_seconds":      round(now - data["last_active"]),
                    "age_seconds":       round(now - data["created_at"]),
                    "documents_indexed": data["pipeline"].stats()["documents"],
                }
                for sid, data in self._sessions.items()
            ]

    # ── Cleanup ───────────────────────────────

    def _cleanup_expired(self):
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, data in self._sessions.items()
                if (now - data["last_active"] > SESSION_INACTIVITY_TIMEOUT)
                or (now - data["created_at"] > SESSION_MAX_LIFETIME)
            ]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            print(f"[SessionManager] Cleaned up {len(expired)} expired session(s).")

    def _start_cleanup_thread(self):
        def loop():
            while True:
                time.sleep(CLEANUP_INTERVAL)
                self._cleanup_expired()

        t = threading.Thread(target=loop, daemon=True)
        t.start()


# Single shared session manager for the whole app
session_manager = SessionManager()