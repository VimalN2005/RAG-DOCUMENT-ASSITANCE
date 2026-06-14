# 🔍 RAG Document Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you chat with your documents using natural language. Upload PDFs, TXT, or DOCX files and instantly get accurate, source-grounded answers powered by FAISS vector search and Groq's LLaMA3.

![Tech Stack](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## ✨ Features

- **Multi-format ingestion** — PDF, TXT, DOCX support via LangChain loaders
- **Semantic chunking** — Recursive text splitter with configurable chunk size & overlap
- **FAISS vector store** — Flat-L2 index with persistence across sessions
- **Sentence-Transformers embeddings** — `all-MiniLM-L6-v2` (runs CPU-only, no GPU needed)
- **Groq LLaMA3 LLM** — Free, blazing-fast inference (swap to OpenAI/Ollama easily)
- **Source citations** — Every answer shows which document it came from
- **Drag & drop UI** — Clean React frontend with real-time stats
- **Docker Compose** — One command to run the full stack

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[React Frontend] ──POST /chat──► [FastAPI Backend]
                                       │
                    ┌──────────────────┤
                    │                  │
              Embed Query          RAG Pipeline
                    │                  │
                    ▼                  ▼
              [sentence-        [FAISS Index]
               transformers]         │
                    │           Top-K Chunks
                    └──────────────────┤
                                       ▼
                               [Groq LLaMA3]
                                       │
                               Grounded Answer
                                       │
                    ◄──────────────────┘
              [React Frontend]
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Free [Groq API key](https://console.groq.com) (takes 30 seconds)

### 1. Clone the repo
```bash
git clone https://github.com/VimalN2005/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY

python main.py
# Backend running at http://localhost:8000
```

### 3. Frontend setup (new terminal)
```bash
cd frontend
npm install
npm run dev
# Frontend running at http://localhost:5173
```

### 4. Or run with Docker
```bash
cp backend/.env.example backend/.env
# Add GROQ_API_KEY in backend/.env

docker-compose up --build
# App available at http://localhost:5173
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Health check |
| `POST` | `/upload` | Upload & index a document |
| `POST` | `/chat`  | Ask a question (RAG + LLM) |
| `GET`  | `/stats` | Index statistics |
| `DELETE` | `/reset` | Clear the vector index |

### Chat Request
```json
POST /chat
{
  "question": "What are the key findings in the report?",
  "top_k": 5,
  "model": "llama3-8b-8192"
}
```

### Chat Response
```json
{
  "answer": "According to the Q3 report, the key findings are...",
  "sources": ["Q3_Report_2024.pdf"],
  "model": "llama3-8b-8192"
}
```

---

## ⚙️ Configuration

| Variable | File | Default | Description |
|----------|------|---------|-------------|
| `GROQ_API_KEY` | `backend/.env` | — | Groq API key (required) |
| `EMBED_MODEL_NAME` | `rag_pipeline.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `rag_pipeline.py` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `rag_pipeline.py` | `64` | Overlap between chunks |

### Swap LLM Provider
To use **OpenAI** instead of Groq, replace in `main.py`:
```python
# from groq import Groq
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Change model to "gpt-4o-mini" or "gpt-3.5-turbo"
```

To use **Ollama** (fully local):
```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# model = "llama3"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| Vector DB | FAISS (CPU) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Document parsing | LangChain Community Loaders |
| LLM | Groq LLaMA3-8B (or OpenAI/Ollama) |
| Frontend | React 18 + Vite |
| Containerization | Docker Compose |

---

## 📁 Project Structure

```
rag-document-assistant/
├── backend/
│   ├── main.py              # FastAPI server & endpoints
│   ├── rag_pipeline.py      # Core RAG logic (embed, index, retrieve)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 🔮 Roadmap

- [ ] Multi-document conversation memory
- [ ] Reranking with cross-encoders (BGE Reranker)
- [ ] Streaming responses (SSE)
- [ ] User authentication & document namespacing
- [ ] Hybrid search (BM25 + dense vectors)
- [ ] LangGraph multi-agent pipeline

---

## 👤 Author

**Vimal Sahani**
- GitHub: [@VimalN2005](https://github.com/VimalN2005)
- Email: vimalsahani2005@gmail.com
- B.Tech Information Technology, IIIT Bhopal (2023–2027)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
