# RAG PDF Q&A Application

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, index them into a PostgreSQL vector store (`pgvector`), and ask natural language questions through an interactive web interface.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, LangChain, LangGraph
- **Frontend:** Node.js / React (`frontend/`)
- **Database:** PostgreSQL 16 with `pgvector` (`docker-compose.yml`)
- **Embeddings:** Ollama (`nomic-embed-text`)
- **LLM Inference:** Groq API (`ChatGroq`) or local Ollama (`ChatOllama`)

---

## Project Structure

```text
Rag-Application/
├── backend/                # RAG pipeline, LangGraph workflow, ingestion, and config
│   ├── config.py           # Environment variable configuration
│   ├── graph.py            # LangGraph agent & LLM routing (Groq / Ollama)
│   ├── ingest.py           # PDF chunking & pgvector indexing
│   ├── main.py             # API route definitions
│   ├── prompts.py          # System and extraction prompts
│   ├── requirements.txt    # Python dependencies
│   └── stream_handler.py   # Streaming response handler
├── frontend/               # Web UI for PDF uploads and Q&A chat
├── main.py                 # FastAPI application entry point
├── docker-compose.yml      # PostgreSQL + pgvector container setup
├── Dockerfile              # Container build instructions
└── .env.example            # Template for environment variables
```

---

## Prerequisites

Before starting, ensure you have the following installed on your machine:

1. **Python 3.10+**
2. **Node.js (v18+) & npm**
3. **Docker & Docker Compose** (for PostgreSQL with `pgvector`)
4. **Ollama** ([Download Ollama](https://ollama.com/)) — required for generating local document embeddings (`nomic-embed-text`)
5. **Groq API Key** *(Optional, Recommended)* — free API key from [Groq Console](https://console.groq.com) for fast cloud LLM inference

---

## Step-by-Step Setup & Startup Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/shruti-bhere/Rag-Application.git](https://github.com/shruti-bhere/Rag-Application.git)
cd Rag-Application
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory from `.env.example`:

```bash
cp .env.example .env
```

Open `.env` and configure your LLM and database settings:

```env
# Database (matches docker-compose.yml)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
COLLECTION_NAME=pdf_documents

# Ollama (required for embeddings; also used for chat if GROQ_API_KEY is not set)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Groq (Recommended for fast inference without downloading large local LLMs)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Retrieval & Extraction Tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=5
EXTRACTOR_MAX_CONTEXT_CHARS=5000
EXTRACTOR_NUM_PREDICT=1024
```

> **Notes on `.env` Configuration:**
> - **Reasoning Models (e.g., `openai/gpt-oss-20b`):** Keep `EXTRACTOR_NUM_PREDICT=1024` (or higher) so the model has enough token budget to finish reasoning and output the final answer.
> - **Port Conflict on `5432`:** If PostgreSQL is already running locally on port `5432`, update the host port in `docker-compose.yml` to `"5433:5432"` and update `DATABASE_URL` in `.env` to `postgresql://postgres:postgres@localhost:5433/ragdb`.

### 3. Pull Required Ollama Models

Start Ollama and pull the embedding model (and chat model if not using Groq):

```bash
# Required for document embeddings
ollama pull nomic-embed-text

# Optional: Only needed if you leave GROQ_API_KEY blank
ollama pull llama3.2
```

### 4. Start the PostgreSQL (`pgvector`) Database

Start the database container in detached mode:

```bash
docker compose up -d
```

Verify the container is healthy:

```bash
docker ps
```

### 5. Start the Backend Server

Create and activate a Python virtual environment, install dependencies, and run FastAPI:

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn main:app --reload
```

The backend API will run at **`http://127.0.0.1:8000`**.

### 6. Start the Frontend

Open a new terminal window, navigate to the `frontend/` folder, install packages, and start the development server:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed in your terminal (typically **`http://localhost:5173`** or **`http://localhost:3000`**) in your browser.

---

## Troubleshooting

- **`Bind for 0.0.0.0:5432 failed: port is already allocated`:** Another Postgres instance is using port `5432`. Stop your local Postgres service (`brew services stop postgresql`) or map Docker to port `5433` in both `docker-compose.yml` and `.env`.
- **`model_not_found` (404) from Groq:** Ensure `GROQ_MODEL` in `.env` matches an active model ID enabled on your Groq account (check available models at `https://api.groq.com/openai/v1/models`).
- **Empty responses from the assistant:** If using a reasoning model on Groq, increase `EXTRACTOR_NUM_PREDICT` in `.env` to `1024` and restart Uvicorn.
