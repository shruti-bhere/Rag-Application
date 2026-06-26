"""
FastAPI app: PDF upload, ingest, and streaming RAG Q&A.
Uses Ollama for embeddings and LLM (no OpenAI API key required).
"""
import logging
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import DATABASE_URL
from backend.graph import get_rag_agent
from backend.ingest import (
    DatabaseUnreachableError,
    clear_vector_store,
    ingest_pdf,
    wait_for_db,
)
from backend.stream_handler import stream_agent_response

logger = logging.getLogger(__name__)


def _postgres_url():
    """URL for the default 'postgres' database (to create ragdb if missing)."""
    u = urlparse(DATABASE_URL)
    return urlunparse((u.scheme, u.netloc, "/postgres", u.params, u.query, u.fragment))


def ensure_database():
    """Create ragdb if it does not exist."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    conn = psycopg2.connect(_postgres_url())
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'ragdb'")
    if not cur.fetchone():
        cur.execute('CREATE DATABASE ragdb')
    cur.close()
    conn.close()


def ensure_pgvector():
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.close()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Python executable (verify .venv): {sys.executable}", flush=True)
    try:
        wait_for_db()
        ensure_database()
        ensure_pgvector()
    except Exception as e:
        logger.warning("Startup DB init: %s (app will retry on first upload/ask)", e)
    yield
    pass


app = FastAPI(title="RAG PDF Q&A", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = get_rag_agent()
    return _agent


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_dir = None
    temp_path = None
    upload_id = uuid.uuid4().hex
    try:
        logger.info("upload: start upload_id=%s filename=%s", upload_id, file.filename)
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        temp_dir = tempfile.mkdtemp()
        safe_name = os.path.basename(file.filename or "upload.pdf").replace("..", "_")
        if not safe_name.lower().endswith(".pdf"):
            safe_name = "upload.pdf"
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        temp_path = os.path.join(temp_dir, unique_name)
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
        logger.info("upload: saved to temp path upload_id=%s path=%s", upload_id, temp_path)

        logger.info("upload: clearing previous vector store upload_id=%s", upload_id)
        clear_vector_store()

        logger.info("upload: starting ingest (load -> split -> index) upload_id=%s", upload_id)
        count = ingest_pdf(temp_path, file.filename)
        logger.info("upload: success upload_id=%s chunks_indexed=%s", upload_id, count)
        thread_id = uuid.uuid4().hex
        return {
            "filename": file.filename,
            "chunks_indexed": count,
            "thread_id": thread_id,
        }
    except HTTPException:
        raise
    except DatabaseUnreachableError as e:
        logger.warning("upload: database unreachable upload_id=%s", upload_id)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e)
        if "corrupt" in msg.lower() or "unreadable" in msg.lower():
            raise HTTPException(status_code=400, detail=msg) from e
        if "database" in msg.lower() or "unreachable" in msg.lower() or "indexing failed" in msg.lower():
            raise HTTPException(status_code=400, detail=msg) from e
        logger.exception("upload: failed upload_id=%s", upload_id)
        raise HTTPException(status_code=500, detail=msg) from e
    except ModuleNotFoundError as e:
        if "fitz" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="PyMuPDF (fitz) not installed. Install with: pip install pymupdf",
            ) from e
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("upload: failed upload_id=%s step=see_traceback", upload_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


@app.get("/health")
async def health():
    return {"status": "ok"}


from pydantic import BaseModel


class AskBody(BaseModel):
    question: str


@app.post("/ask")
async def ask(body: AskBody):
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    agent = get_agent()

    async def event_stream():
        async for chunk in stream_agent_response(agent, question):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
