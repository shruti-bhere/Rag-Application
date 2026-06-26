# RAG PDF Q&A

Upload any PDF, index it into Postgres + pgvector, and chat with it in natural language.  
Backend: FastAPI + LangGraph/LangChain (Groq or Ollama LLM). Frontend: React + Vite + Framer Motion.

For internal architecture details, see `PROJECT_FLOW.md`. This README focuses only on how to start and use the app.

---

## 1. Prerequisites

- **Docker** – Postgres + pgvector.
- **Python 3.10+** – backend.
- **Node 18+** – frontend.
- LLM backend:
  - **Groq** (recommended): free API key from `https://console.groq.com`, or
  - **Ollama**: installed and running from `https://ollama.ai`.

---

## 2. Configure `.env`

From the project root:

```bash
cp .env.example .env
```

Then edit `.env`:

- Database (use default if you keep `docker-compose.yml` as is):

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
```

- Groq (optional but recommended for fast LLM responses):

```env
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

- Ollama embeddings (install the model once):

```bash
ollama pull nomic-embed-text
```

Other values in `.env.example` (chunk size, overlap, extraction limits) can stay as‑is to start.

---

## 3. Run backend

From the project root:

```bash
docker-compose up -d                 # start Postgres + pgvector

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend exposes:

- `POST /upload` – upload and index a PDF.
- `POST /ask` – ask a question; answer streams via Server‑Sent Events (SSE).
- `GET /health` – health check.

---

## 4. Run frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 5. Use the app

1. **Upload a PDF**
   - Click **UPLOAD PDF** (or drop a PDF) in the UI.
   - The backend ingests and indexes the document; you’ll see `Indexed N chunks from <filename>`.
2. **Ask questions in natural language**
   - Examples:
     - “What is the address?”
     - “Extract all technical skills.”
     - “Summarize section 3.”
3. **Read streaming answers**
   - Answers appear token‑by‑token in the chat, extracted verbatim from the uploaded PDF.

# RAG PDF Q&A Application

A retrieval‑augmented generation (RAG) app for PDF documents: upload a PDF, index it into a Postgres + pgvector store, and ask questions with **streaming, verbatim answers** over HTTP SSE.  
Backend is FastAPI + LangGraph/LangChain; frontend is React + Vite + Framer Motion with a cyberpunk black/neon‑green UI.  
LLM backend can be either:

- **Groq** (`ChatGroq`, e.g. `llama-3.1-8b-instant`) – instant, free API, or
- **Ollama** (`ChatOllama`) – local model server (default fallback when no `GROQ_API_KEY` is set).

## Features

- **PDF ingest**: PyMuPDFLoader loads PDFs; RecursiveCharacterTextSplitter (chunk 1000, overlap 100) chunks text; Ollama embeddings; PGVector (PostgreSQL + pgvector) for storage.
- **Parent‑style retrieval**: Optional small child chunks for vector search with full parent context in metadata for better extraction accuracy.
- **Search + retrieval graph**: LangGraph state machine (`graph.py`) with:
  - **Search node** – builds one or more semantic queries (or uses the user text directly for short/field questions).
  - **Retrieval node** – runs parallel vector searches + optional window expansion and wraps results in `[source] ... [/source]`.
  - **Extractor node** – strict SYSTEM_RAG prompt; returns a **verbatim substring** of the context or `Not found in document.`.
- **Fast paths (no LLM)**: generic `Label: value` / `Label\nvalue` extraction and section extraction under detected headings (works for any field name, any PDF).
- **Streaming**: Final extraction is streamed token‑by‑token via Server‑Sent Events (SSE) so the UI shows live typing.
- **Cyberpunk UI**: React + Vite + Framer Motion with pure black background, neon green accents, animated grid, drag‑and‑drop PDF upload, and streaming chat.

## Prerequisites

- **Docker** (for Postgres + pgvector)
- **Python 3.10+** (backend)
- **Node 18+** (frontend)
- At least one LLM backend:
  - **Groq** account + API key (recommended for instant responses), or
  - **Ollama** installed and running ([ollama.ai](https://ollama.ai)) for local models.
- **Python 3.10+** (backend)
- **Node 18+** (frontend)

### Recommended: Groq (LLM) + Ollama (embeddings)

- Groq:
  - Get a free API key from `https://console.groq.com`.
  - Default model: `llama-3.1-8b-instant` (configurable via `GROQ_MODEL`).
- Ollama:
  - Used for **embeddings** (and as fallback LLM if `GROQ_API_KEY` is not set).
  - Install models:

    ```bash
    # LLM for local fallback (optional)
    ollama pull llama3.2

    # Embeddings (for vector search)
    ollama pull nomic-embed-text
    ```

## Quick start

### 1. Start Postgres (pgvector)

```bash
docker-compose up -d
```

### 2. Backend (FastAPI + LangGraph)

From the **project root** (`rag-application`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env       # edit .env to set DATABASE_URL, GROQ_API_KEY, etc.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or from `backend/`: `uvicorn main:app --reload --host 0.0.0.0 --port 8000` (after `pip install -r requirements.txt` there).

### 3. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Upload a PDF, then ask questions; answers stream in the chat.

## Configuration

Copy `.env.example` to `.env` in the **project root** and adjust if needed:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Postgres connection string (default: `postgresql://postgres:postgres@localhost:5432/ragdb`) |
| `COLLECTION_NAME` | Vector collection name |
| `OLLAMA_BASE_URL` | Ollama API URL (for embeddings and optional fallback chat) |
| `OLLAMA_MODEL` | Fallback chat model (e.g. `llama3.2`) |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model (e.g. `nomic-embed-text`) |
| `GROQ_API_KEY` | Groq API key; when set, Groq is used for all LLM calls |
| `GROQ_MODEL` | Groq chat model (default `llama-3.1-8b-instant`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Text splitter settings |
| `TOP_K_RETRIEVAL` | Number of chunks to retrieve per query |

## Project layout

- **backend/**
  - `main.py` – FastAPI app: `/upload`, `/ask` (streaming SSE), `/health`; DB bootstrap.
  - `config.py` – Settings from env (`DATABASE_URL`, `GROQ_API_KEY`, models, chunking, limits).
  - `prompts.py` – Generic, non‑hardcoded prompts for search/extraction.
  - `ingest.py` – PDF load, chunk (RecursiveCharacterTextSplitter), embed (Ollama), PGVector index.
  - `graph.py` – LangGraph RAG pipeline:
    - search node, retrieval node, extractor node, fast‑path extractors.
    - switches between `ChatGroq` and `ChatOllama` based on `GROQ_API_KEY`.
  - `stream_handler.py` – Runs search + retrieval, then streams extractor tokens over SSE.
- **frontend/**
  - React + Vite app:
    - `src/api.js` – calls `/upload` and `/ask` (SSE) on the backend.
    - `src/App.jsx` – upload dropzone, cyberpunk upload button, streaming chat UI.
    - `src/App.css`, `src/index.css` – neon black/green theme and animations.
- **docker-compose.yml** – Postgres with pgvector.
- **.env.example** – Example env vars for root `.env`.

## API

- `POST /upload`
  - Body: form‑data with `file` (PDF).
  - Effect: clears previous index, ingests the new PDF (load → chunk → embed → upsert).
  - Response: `{ "filename": str, "chunks_indexed": int, "thread_id": str }`.
- `POST /ask`
  - Body: JSON `{ "question": "..." }`.
  - Effect: runs search + retrieval once, then streams extractor tokens.
  - Response: Server‑Sent Events:
    - `data: {"type":"token","content":"..."}` for each text chunk,
    - `data: {"type":"error","content":"..."}` on error,
    - `data: [DONE]` at end.
- `GET /health` – Health check (`{"status":"ok"}`).

## Tech stack

- **Backend**
  - FastAPI, LangChain, LangGraph, langchain‑community (PGVector), langchain‑ollama, langchain‑groq
  - pypdf / PyMuPDF, pgvector, psycopg2‑binary
- **Frontend**
  - React 18, Vite 5, Framer Motion
  - Custom CSS (pure black + neon green), drag‑and‑drop + streaming chat
- **Data / Models**
  - PostgreSQL + pgvector
  - Ollama (embeddings + optional local LLM)
  - Groq (ChatGroq, recommended for instant responses)

For a more detailed, step‑by‑step explanation of which functions run first and how the whole flow works (upload → ingest → search → retrieval → streaming extraction), see `PROJECT_FLOW.md`.
