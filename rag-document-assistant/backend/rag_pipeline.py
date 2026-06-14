"""
RAG Pipeline — Core document ingestion, chunking, embedding, and retrieval.
"""

import os
import json
import hashlib
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
INDEX_PATH       = "data/faiss.index"
META_PATH        = "data/metadata.json"
UPLOAD_DIR       = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Load & parse documents (PDF / TXT / DOCX)
      2. Chunk with overlap
      3. Embed with sentence-transformers
      4. Store in FAISS flat-L2 index
      5. Retrieve top-k chunks for a query
    """

    def __init__(self):
        print("[RAG] Loading embedding model …")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Load or create FAISS index + metadata
        if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
            print("[RAG] Loading existing index …")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH) as f:
                self.metadata: List[Dict] = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata: List[Dict] = []

    # ── Document Loading ──────────────────────

    def _load_file(self, filepath: str) -> List[str]:
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

    # ── Ingest ────────────────────────────────

    def ingest(self, filepath: str, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Parse, chunk, embed, and index a document."""
        source_name = source_name or Path(filepath).name
        file_hash = hashlib.md5(open(filepath, "rb").read()).hexdigest()

        # Skip if already indexed
        existing = [m for m in self.metadata if m.get("hash") == file_hash]
        if existing:
            return {"status": "skipped", "reason": "already indexed", "source": source_name}

        raw_pages = self._load_file(filepath)
        full_text = "\n\n".join(raw_pages)
        chunks = self.splitter.split_text(full_text)

        if not chunks:
            return {"status": "error", "reason": "no content extracted"}

        vectors = self.embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
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

        self._save()
        return {"status": "ok", "chunks": len(chunks), "source": source_name}

    # ── Retrieve ──────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k most relevant chunks for a query."""
        if self.index.ntotal == 0:
            return []

        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
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

    # ── Persistence ───────────────────────────

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)

    # ── Stats ─────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        sources = list({m["source"] for m in self.metadata})
        return {
            "total_chunks": self.index.ntotal,
            "documents":    len(sources),
            "sources":      sources,
        }
