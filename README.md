# 🎓 Athena Institute of Technology — AI Chatbot

A production-style **RAG (Retrieval-Augmented Generation) chatbot** for Athena Institute of Technology. Built with LangChain, LangGraph, ChromaDB, Redis, OpenAI, and FastAPI.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technologies](#3-technologies)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [Environment Variables](#6-environment-variables)
7. [Redis Setup](#7-redis-setup)
8. [ChromaDB Setup](#8-chromadb-setup)
9. [OpenAI / Azure OpenAI Setup](#9-openai--azure-openai-setup)
10. [Document Ingestion](#10-document-ingestion)
11. [Running the Backend](#11-running-the-backend)
12. [Running the Frontend](#12-running-the-frontend)
13. [API Endpoints](#13-api-endpoints)
14. [RAG Flow](#14-rag-flow)
15. [LangGraph Workflow](#15-langgraph-workflow)
16. [Docker](#16-docker)
17. [Testing](#17-testing)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Project Overview

The Athena Institute AI Chatbot answers questions about:

- Undergraduate and postgraduate programs
- Admission eligibility and procedures
- Fee structures and scholarships
- Campus facilities (library, labs, sports, etc.)
- Hostel and accommodation
- Placement statistics and recruiters
- Faculty information
- Academic calendar and events
- Student policies and regulations
- Contact information and FAQs

The chatbot uses **RAG architecture** to retrieve information from official institute documents before generating answers. It **never fabricates** institutional information — if the answer is not in the knowledge base, it says so honestly.

---

## 2. Architecture

```
User
  │
  ▼
HTML / JavaScript Chat UI (frontend/)
  │
  ▼ POST /api/chat
FastAPI (app/main.py)
  │
  ▼
LangGraph Workflow (app/chatbot/graph.py)
  │
  ├─────────────────────┐
  ▼                     ▼
Redis                 ChromaDB
(Session Memory)      (Vector Knowledge Base)
  │                     │
  │                     ▼
  │             Relevant Document Chunks
  │                     │
  └──────────┬──────────┘
             ▼
        LangChain + OpenAI LLM
             │
             ▼
       Final Answer + Sources
             │
             ▼
          FastAPI
             │
             ▼
          Frontend
```

---

## 3. Technologies

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI, Python 3.12 |
| **RAG Orchestration** | LangGraph |
| **RAG Components** | LangChain |
| **Vector Database** | ChromaDB (persistent) |
| **Session Memory** | Redis |
| **LLM** | OpenAI GPT-4o (or Azure OpenAI) |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Config** | Pydantic Settings |
| **Testing** | pytest, pytest-asyncio |
| **Containerization** | Docker, Docker Compose |

---

## 4. Project Structure

```
athena-ai-chatbot/
│
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── api/
│   │   └── chat.py                # /api/chat and /health endpoints
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   └── logging.py             # Structured logging
│   ├── chatbot/
│   │   ├── graph.py               # LangGraph workflow
│   │   ├── state.py               # ChatState TypedDict
│   │   ├── nodes.py               # LangGraph node functions
│   │   └── prompts.py             # Prompt templates
│   ├── rag/
│   │   ├── embeddings.py          # Embeddings helper
│   │   ├── retriever.py           # Semantic retrieval
│   │   ├── vectorstore.py         # ChromaDB wrapper
│   │   └── ingestion.py           # Document ingestion pipeline
│   ├── memory/
│   │   └── redis_manager.py       # Redis session manager
│   ├── llm/
│   │   └── provider.py            # LLM/Embeddings factory
│   └── schemas/
│       └── chat.py                # Pydantic request/response schemas
│
├── frontend/
│   ├── index.html                 # Single-page application
│   ├── css/
│   │   └── style.css              # Full design system
│   └── js/
│       └── chatbot.js             # Chat UI logic
│
├── data/
│   └── documents/                 # Place your documents here
│       └── athena_institute_knowledge_base.md
│
├── chroma_data/                   # ChromaDB persistence (auto-created)
├── prompts/
│   └── chatbot_prompt.txt         # System prompt
├── scripts/
│   └── ingest.py                  # Document ingestion CLI
├── tests/
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_redis.py
│   └── test_retrieval.py
│
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 5. Installation

### Prerequisites

- Python 3.12+
- Redis server running locally (or via Docker)
- OpenAI API key (or Azure OpenAI credentials)

### Step 1: Clone / navigate to the project

```bash
cd athena-ai-chatbot
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure environment variables

```bash
copy .env.example .env
```

Edit `.env` with your actual values (see [Section 6](#6-environment-variables)).

---

## 6. Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```env
# Application
APP_ENV=development
DEBUG=false

# OpenAI (use one of standard or Azure)
OPENAI_API_KEY=sk-...your-key-here...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Azure OpenAI (optional, leave blank if using standard OpenAI)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=

# Redis
REDIS_URL=redis://localhost:6379
REDIS_SESSION_TTL=3600
REDIS_MAX_HISTORY=20

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_data
CHROMA_COLLECTION_NAME=athena_knowledge

# RAG Configuration
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.70
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# LLM
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024
LLM_TIMEOUT=60
```

> ⚠️ **Never commit your `.env` file.** It is already in `.gitignore`.

---

## 7. Redis Setup

### Option A: Local Redis (recommended for development)

**Windows** — Using Memurai (Redis-compatible):
```
Download from https://www.memurai.com/
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

**Verify Redis is running:**
```bash
redis-cli ping
# Expected output: PONG
```

### Option B: Docker

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

---

## 8. ChromaDB Setup

ChromaDB runs **embedded/persistently** — no separate server required.

The database is automatically created at `./chroma_data` when you first run the ingestion script or start the server.

To change the persistence directory:
```env
CHROMA_PERSIST_DIRECTORY=./my-chroma-data
```

---

## 9. OpenAI / Azure OpenAI Setup

### Standard OpenAI

1. Get your API key from https://platform.openai.com/api-keys
2. Set in `.env`:
   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   ```

### Azure OpenAI

1. Create an Azure OpenAI resource in the Azure portal.
2. Deploy a chat model (e.g., `gpt-4o`) and an embedding model (e.g., `text-embedding-ada-002`).
3. Set in `.env`:
   ```env
   AZURE_OPENAI_API_KEY=your-azure-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-01
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
   ```

> The system automatically switches to Azure when `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are both set.

---

## 10. Document Ingestion

### Add your documents

Place your institute documents in:
```
data/documents/
```

Supported formats:
- `.pdf` — Admission brochures, fee structures, policies
- `.docx` — FAQs, program guides
- `.txt` — Plain text documents
- `.md` — Markdown documents

A sample knowledge base (`athena_institute_knowledge_base.md`) is included to get you started immediately.

### Run the ingestion script

```bash
python scripts/ingest.py
```

**With options:**
```bash
# Custom documents directory
python scripts/ingest.py --data-dir ./data/documents

# Force re-ingest all documents (skips deduplication)
python scripts/ingest.py --force

# Enable verbose logging
python scripts/ingest.py --verbose
```

The script will:
1. Load all supported documents from the directory
2. Split them into semantic chunks (800 chars, 150 overlap)
3. Generate embeddings using OpenAI
4. Store in ChromaDB with metadata (source file, page number)
5. Skip chunks already in ChromaDB (deduplication)
6. Print a summary of new chunks added

---

## 11. Running the Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

Swagger docs (development only): `http://localhost:8000/docs`

---

## 12. Running the Frontend

The frontend is **served by FastAPI** as static files. Simply open:

```
http://localhost:8000
```

No separate frontend server is needed.

---

## 13. API Endpoints

### `POST /api/chat`

Send a message to the chatbot.

**Request:**
```json
{
  "session_id": "sess_abc123",
  "message": "What courses are available at Athena Institute?"
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "answer": "Athena Institute of Technology offers the following programs...",
  "sources": [
    {
      "source": "athena_institute_knowledge_base.md",
      "page": null,
      "chunk_index": 5
    }
  ],
  "rewritten_query": "What courses are available at Athena Institute of Technology?"
}
```

### `GET /health`

Check the health of all services.

**Response:**
```json
{
  "status": "healthy",
  "redis": "connected",
  "chromadb": "connected (247 docs)",
  "llm": "configured",
  "version": "1.0.0"
}
```

---

## 14. RAG Flow

```
1. User sends message  →  POST /api/chat
2. FastAPI validates input (length, required fields)
3. LangGraph workflow starts with initial state
4. Node: load_session_history — retrieves conversation from Redis
5. Node: rewrite_query — uses LLM to make query self-contained
6. Node: retrieve_context — semantic search in ChromaDB
   - Converts query to embedding
   - Finds top-K similar chunks
   - Filters by similarity threshold (0.70)
7. Conditional routing:
   - Relevant docs found → generate_answer
   - No relevant docs → fallback_response
8. Node: generate_answer — LLM generates answer using retrieved context
9. Node: update_session — saves exchange to Redis with TTL
10. FastAPI returns answer + sources
11. Frontend displays response with source citations
```

---

## 15. LangGraph Workflow

```
START
  │
  ▼
load_session_history
  │ (loads Redis conversation history)
  ▼
rewrite_query
  │ (rewrites query to be self-contained using chat history)
  ▼
retrieve_context
  │ (semantic search in ChromaDB, filters by threshold)
  ▼
[is_relevant?]
  ├── True  ──► generate_answer ──► update_session ──► END
  └── False ──► fallback_response ──► update_session ──► END
```

**ChatState (TypedDict):**
```python
class ChatState(TypedDict):
    session_id: str
    question: str
    chat_history: list[dict]
    rewritten_query: str
    retrieved_documents: list
    context: str
    answer: str
    sources: list[dict]
    is_relevant: bool
    error: Optional[str]
```

---

## 16. Docker

### Full stack (recommended)

```bash
# Copy and configure environment
copy .env.example .env
# Edit .env with your API keys

# Build and start services
docker-compose up --build

# In a separate terminal, run ingestion
docker-compose exec api python scripts/ingest.py
```

Services:
- API: `http://localhost:8000`
- Redis: `localhost:6379`

### Stop services

```bash
docker-compose down

# Remove volumes (clears Redis data)
docker-compose down -v
```

---

## 17. Testing

```bash
# Run all tests
pytest

# With verbose output
pytest -v

# Run specific test file
pytest tests/test_redis.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=app --cov-report=html
```

---

## 18. Troubleshooting

### ❌ "Redis connection refused"
- Ensure Redis is running: `redis-cli ping`
- Check `REDIS_URL` in `.env`
- If using Docker: `docker-compose up redis`

### ❌ "No API key found"
- Ensure `OPENAI_API_KEY` is set in `.env`
- For Azure: ensure all `AZURE_OPENAI_*` variables are set

### ❌ "No documents in ChromaDB / no relevant docs found"
- Run the ingestion script: `python scripts/ingest.py`
- Check the data directory has documents: `ls data/documents/`
- Lower `SIMILARITY_THRESHOLD` in `.env` (e.g., 0.50)

### ❌ "ChromaDB collection is empty after ingestion"
- Verify `CHROMA_PERSIST_DIRECTORY` path is writable
- Check ingestion logs for errors
- Try `python scripts/ingest.py --force --verbose`

### ❌ "LLM timeout"
- Increase `LLM_TIMEOUT` in `.env`
- Try a faster model: `OPENAI_MODEL=gpt-4o-mini`

### ❌ Frontend shows "Could not connect to server"
- Ensure the backend is running: `uvicorn app.main:app --reload`
- Check `http://localhost:8000/health`

### 🔍 Enable debug logging
```env
DEBUG=true
```

Then restart the server to see detailed log output.

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Edit .env: add OPENAI_API_KEY

# 3. Start Redis
redis-server   # or: docker run -d -p 6379:6379 redis:7-alpine

# 4. Ingest documents
python scripts/ingest.py

# 5. Start the server
uvicorn app.main:app --reload

# 6. Open the chatbot
# → http://localhost:8000
```
