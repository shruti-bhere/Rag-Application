"""
PDF loading with PyMuPDFLoader (primary) or PyPDFLoader (fallback), chunking with
RecursiveCharacterTextSplitter, and indexing into PGVector (Ollama embeddings).
Supports ParentDocumentRetriever-style storage.
"""
import hashlib
import logging
import os
import time
from typing import List

logger = logging.getLogger(__name__)

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATABASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    USE_PARENT_DOCUMENT_RETRIEVER,
)

# Wait for Postgres/PGVector to be ready before first connection (seconds).
DB_WAIT_TIMEOUT = int(os.getenv("DB_WAIT_TIMEOUT", "60"))
DB_WAIT_INTERVAL = float(os.getenv("DB_WAIT_INTERVAL", "2.0"))


class DatabaseUnreachableError(RuntimeError):
    """Raised when the database cannot be reached after retries."""
    pass


def _load_pdf_pymupdf(file_path: str) -> List[Document]:
    """Load PDF with PyMuPDFLoader (requires pymupdf; uses fitz internally)."""
    from langchain_community.document_loaders import PyMuPDFLoader
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def _load_pdf_pypdf(file_path: str) -> List[Document]:
    """Load PDF with PyPDFLoader (pypdf) as fallback."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_pdf_documents(file_path: str, source_label: str) -> List[Document]:
    """Load PDF: try PyMuPDFLoader first, fall back to PyPDFLoader (pypdf) if pymupdf/fitz is missing or fails."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    docs = None
    first_error = None
    try:
        docs = _load_pdf_pymupdf(file_path)
    except Exception as e:
        first_error = e
        try:
            docs = _load_pdf_pypdf(file_path)
        except Exception as fallback_e:
            raise RuntimeError(
                f"PDF loading failed (PyMuPDF: {first_error!s}; pypdf fallback: {fallback_e!s})"
            ) from fallback_e

    if not docs:
        return []
    for d in docs:
        d.metadata["source"] = source_label
    return docs


_vector_store_cache = None
_db_ready = False


def _check_db_connection() -> bool:
    """Try to connect to the database. Returns True if successful."""
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        return True
    except Exception:
        return False


def wait_for_db(timeout_seconds: int = DB_WAIT_TIMEOUT, interval: float = DB_WAIT_INTERVAL) -> None:
    """Block until the database is reachable or timeout. Raises DatabaseUnreachableError on timeout."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _check_db_connection():
            global _db_ready
            _db_ready = True
            logger.info("Database connection ready.")
            return
        logger.debug("Waiting for database (retrying in %s s)...", interval)
        time.sleep(interval)
    raise DatabaseUnreachableError(
        f"Database unreachable after {timeout_seconds}s. Check that Postgres/PGVector is running and DATABASE_URL is correct."
    )


def verify_db_connection() -> None:
    """Ensure the database is reachable. Raises DatabaseUnreachableError if not."""
    if _db_ready and _vector_store_cache is not None:
        return
    if not _check_db_connection():
        wait_for_db()


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store(embedding=None):
    """Return a single persistent PGVector instance. Lazy-creates store and ensures extension, tables, and collection exist."""
    global _vector_store_cache
    if _vector_store_cache is not None:
        return _vector_store_cache
    wait_for_db()
    if embedding is None:
        embedding = get_embeddings()
    _vector_store_cache = PGVector(
        connection_string=DATABASE_URL,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    # Ensure collection exists (PGVector __init__ already does this; re-call to be robust after delete_collection).
    try:
        _vector_store_cache.create_collection()
    except Exception as e:
        logger.warning("create_collection after init: %s", e)
    return _vector_store_cache


def clear_vector_store() -> None:
    """Clear the PDF collection so a new upload becomes the single source of truth. Recreates an empty collection to avoid 'Collection not found' on next search."""
    try:
        vectorstore = get_vector_store()
        vectorstore.delete_collection()
        logger.info("Vector store collection cleared successfully.")
        vectorstore.create_collection()
        logger.debug("Empty collection recreated for next search/ingest.")
    except Exception as e:
        if "collection" in str(e).lower() and "not found" in str(e).lower():
            logger.info("clear_vector_store: collection already missing, ensuring it exists.")
            try:
                get_vector_store().create_collection()
            except Exception as e2:
                logger.warning("create_collection after clear: %s", e2)
        else:
            logger.warning("clear_vector_store: %s (continuing with ingest)", e)


def _parent_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def index_documents(docs: List[Document]) -> None:
    if not docs:
        return
    embedding = get_embeddings()
    vectorstore = get_vector_store(embedding)
    vectorstore.add_documents(docs)


def ingest_pdf(file_path: str, original_filename: str) -> int:
    """Load PDF, split, and index. Verifies DB before clear/index. Raises RuntimeError/DatabaseUnreachableError on failure."""
    logger.info("ingest_pdf: starting load step for %s", original_filename)
    verify_db_connection()

    try:
        source_id = hashlib.sha256(original_filename.encode()).hexdigest()[:16]
        source_label = f"{original_filename} ({source_id})"
        raw_docs = load_pdf_documents(file_path, source_label)
        logger.info("ingest_pdf: load step ok, pages=%s", len(raw_docs))
    except Exception as e:
        logger.exception("ingest_pdf: document loading failed")
        raise RuntimeError(f"PDF is corrupt or unreadable: {e!s}") from e

    if not raw_docs:
        logger.warning("ingest_pdf: no documents loaded, skipping")
        return 0
    text = "\n\n".join(d.page_content for d in raw_docs)
    if not text.strip():
        logger.warning("ingest_pdf: no text in documents, skipping")
        return 0

    logger.info("ingest_pdf: starting split step")
    try:
        parent_splitter = _parent_splitter()
        parent_chunks = parent_splitter.split_text(text)
        parent_docs = [
            Document(page_content=c, metadata={"source": source_label})
            for c in parent_chunks
        ]
        logger.info("ingest_pdf: split step ok, parent_chunks=%s", len(parent_docs))
    except Exception as e:
        logger.exception("ingest_pdf: chunking failed")
        raise RuntimeError(f"Chunking failed: {e!s}") from e

    if not parent_docs:
        return 0

    logger.info("ingest_pdf: starting indexing step")
    try:
        if USE_PARENT_DOCUMENT_RETRIEVER:
            child_splitter = _child_splitter()
            child_docs: List[Document] = []
            chunk_index = 0
            for parent in parent_docs:
                children = child_splitter.split_text(parent.page_content)
                for c in children:
                    child_docs.append(
                        Document(
                            page_content=c,
                            metadata={
                                "source": parent.metadata.get("source", source_label),
                                "parent_content": parent.page_content,
                                "chunk_index": chunk_index,
                            },
                        )
                    )
                    chunk_index += 1
            index_documents(child_docs)
            logger.info("ingest_pdf: indexing step ok, chunks_indexed=%s", len(child_docs))
            return len(child_docs)
        else:
            docs = [
                Document(
                    page_content=p.page_content,
                    metadata={**p.metadata, "chunk_index": i},
                )
                for i, p in enumerate(parent_docs)
            ]
            index_documents(docs)
            logger.info("ingest_pdf: indexing step ok, chunks_indexed=%s", len(docs))
            return len(docs)
    except DatabaseUnreachableError:
        raise
    except Exception as e:
        logger.exception("ingest_pdf: vectorization or indexing failed")
        raise RuntimeError(
            f"Database unreachable or indexing failed. Check that Postgres is running and reachable: {e!s}"
        ) from e
