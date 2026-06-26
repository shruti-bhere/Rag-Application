import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/ragdb",
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Groq (free API for instant responses). If set, RAG uses ChatGroq instead of Ollama.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "5"))
RETRIEVAL_WINDOW_SIZE = int(os.getenv("RETRIEVAL_WINDOW_SIZE", "2"))
# Comma-separated prefixes for single-field requests (trailing space matters). Set in env; no default list in code.
FIELD_REQUEST_PREFIXES = [p.strip() for p in os.getenv("FIELD_REQUEST_PREFIXES", "").split(",") if p.strip()]
# ParentDocumentRetriever: small chunks for search, full parent for context
USE_PARENT_DOCUMENT_RETRIEVER = os.getenv("USE_PARENT_DOCUMENT_RETRIEVER", "true").lower() in ("1", "true", "yes")
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "300"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# Extraction speed: limit context size and max tokens; configurable so long lists/sections aren't cut off.
EXTRACTOR_MAX_CONTEXT_CHARS = int(os.getenv("EXTRACTOR_MAX_CONTEXT_CHARS", "5000"))
EXTRACTOR_NUM_PREDICT = int(os.getenv("EXTRACTOR_NUM_PREDICT", "120"))
# Optional: use a smaller/faster model for extraction only (set in env to override OLLAMA_MODEL for extractor).
OLLAMA_EXTRACTOR_MODEL = os.getenv("OLLAMA_EXTRACTOR_MODEL", "").strip() or OLLAMA_MODEL
# Section extraction fast path: max lines to return when matching a heading block (no LLM).
SECTION_EXTRACT_MAX_LINES = int(os.getenv("SECTION_EXTRACT_MAX_LINES", "50"))
