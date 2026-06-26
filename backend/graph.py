"""
RAG pipeline built as a LangGraph with specialized nodes:
- Search node: multi-query expansion.
- Retrieval node: parallel vector searches + [source] formatting.
- Extractor node: strict SYSTEM_RAG verbatim extractor.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, TypedDict

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from backend.config import (
    EXTRACTOR_MAX_CONTEXT_CHARS,
    EXTRACTOR_NUM_PREDICT,
    FIELD_REQUEST_PREFIXES,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_EXTRACTOR_MODEL,
    RETRIEVAL_WINDOW_SIZE,
    SECTION_EXTRACT_MAX_LINES,
    TOP_K_RETRIEVAL,
    USE_PARENT_DOCUMENT_RETRIEVER,
)
from backend.ingest import get_vector_store
from backend.prompts import MULTI_QUERY_PROMPT, RAG_PROMPT

NOT_FOUND = "Not found in document."


class AgentState(TypedDict, total=False):
    """LangGraph state passed between nodes."""

    question: str
    search_queries: List[str]
    context: str
    answer: str


_SEARCH_WORD_THRESHOLD = 2
_SEARCH_CHAR_THRESHOLD = 25


def _use_groq() -> bool:
    return bool(GROQ_API_KEY)


def get_llm():
    """Default LLM. Uses Groq when GROQ_API_KEY is set, else Ollama."""
    if _use_groq():
        return ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
        )
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


def get_llm_search():
    """LLM for Search node: minimal tokens, deterministic."""
    if _use_groq():
        return ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
            max_tokens=50,
        )
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=50,
    )


def get_llm_extractor():
    """LLM for Extractor node: model and token limit from config for faster extraction."""
    if _use_groq():
        return ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
            max_tokens=EXTRACTOR_NUM_PREDICT,
        )
    return ChatOllama(
        model=OLLAMA_EXTRACTOR_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=EXTRACTOR_NUM_PREDICT,
    )


def _expand_window(vectorstore, docs, window_size: int = RETRIEVAL_WINDOW_SIZE):
    """For each doc with source and chunk_index, fetch docs with chunk_index in [idx-window_size, idx+window_size]. Returns combined, deduped, sorted by chunk_index."""
    seen = set()
    by_source = {}
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        if not isinstance(meta, dict):
            continue
        src = meta.get("source")
        idx = meta.get("chunk_index")
        if src is None or idx is None:
            continue
        idx = int(idx) if isinstance(idx, (int, float)) else idx
        if src not in by_source:
            by_source[src] = set()
        for delta in range(-window_size, window_size + 1):
            by_source[src].add(max(0, idx + delta))

    if not by_source:
        return docs

    expanded = []
    for source, indices in by_source.items():
        try:
            filter_dict = {"source": source, "chunk_index": {"$in": list(indices)}}
            window_docs = vectorstore.similarity_search(
                " ", k=max(len(indices) + 10, 20), filter=filter_dict
            )
            for d in window_docs:
                meta = getattr(d, "metadata", {}) or {}
                key = (meta.get("source"), meta.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    expanded.append(d)
        except Exception:
            continue

    if not expanded:
        return docs
    expanded.sort(key=lambda d: (str((d.metadata or {}).get("source", "")), (d.metadata or {}).get("chunk_index", 0)))
    return expanded


def _format_docs(docs, exclude_queries=None) -> str:
    """Return only document text wrapped in [source] tags. Drops lines that match generated search queries."""
    if not docs:
        return ""
    exclude = set()
    if exclude_queries:
        for q in exclude_queries:
            if isinstance(q, str) and q.strip():
                exclude.add(q.strip().lower())
    parts = []
    for doc in docs:
        content = doc.page_content
        if USE_PARENT_DOCUMENT_RETRIEVER:
            parent = getattr(doc, "metadata", {}) or {}
            if isinstance(parent, dict) and parent.get("parent_content"):
                content = parent["parent_content"]
        if exclude:
            lines = [ln for ln in content.split("\n") if ln.strip().lower() not in exclude]
            content = "\n".join(lines)
        if not content.strip():
            continue
        parts.append(f"[source]\n{content}\n[/source]")
    return "\n\n---\n\n".join(parts) if parts else ""


_shared_cache: Dict[str, object] | None = None


def _build_shared_components() -> Dict[str, object]:
    """Create and cache shared components (LLMs, vector store, retriever, chains) used by nodes."""
    global _shared_cache
    if _shared_cache is not None:
        return _shared_cache
    llm_search = get_llm_search()
    llm_extractor = get_llm_extractor()
    vectorstore = get_vector_store()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    multi_chain = MULTI_QUERY_PROMPT | llm_search
    extractor_chain = RAG_PROMPT | llm_extractor
    _shared_cache = {
        "vectorstore": vectorstore,
        "base_retriever": base_retriever,
        "multi_chain": multi_chain,
        "extractor_chain": extractor_chain,
    }
    return _shared_cache


def _is_short_query(question: str) -> bool:
    """True if we should skip MULTI_QUERY and use the question as the single search query."""
    q = (question or "").strip()
    if not q:
        return True
    words = q.split()
    return len(words) <= _SEARCH_WORD_THRESHOLD or len(q) <= _SEARCH_CHAR_THRESHOLD


def _extract_field_query(question: str) -> str | None:
    """If question starts with a configured prefix and has a short rest, return that rest as the search term."""
    q = (question or "").strip().rstrip("?").lower()
    if not q or len(q) > 80:
        return None
    for prefix in FIELD_REQUEST_PREFIXES:
        if q.startswith(prefix):
            rest = q[len(prefix):].strip()
            if rest and len(rest.split()) <= 3:
                return rest.strip() or None
    return None


def get_field_query(question: str) -> str | None:
    """Exposed for stream handler: same as _extract_field_query."""
    return _extract_field_query(question)


def _label_value_pattern(field: str) -> str:
    """Generic regex for 'Label: value' or 'Label - value' on same line; label from request (any document)."""
    label = re.escape(field.strip()).replace(" ", r"\s+")
    return rf"(?:{label})\s*[:\-]\s*(.+?)(?=\n|$)"


def _label_next_line_pattern(field: str) -> str:
    """Generic regex for 'Label:' or 'Label -' at end of line with value on next line (common in PDFs)."""
    label = re.escape(field.strip()).replace(" ", r"\s+")
    return rf"(?:{label})\s*[:\-]\s*\n\s*([^\n]+)"


def try_fast_extract(context: str, field_query: str | None) -> str | None:
    """
    If the document has 'Label: value' (same line) or 'Label:\\nvalue' (next line), return the value without calling the LLM.
    Works for any field; no hardcoded names. Returns None if no match so caller falls back to the extractor.
    """
    if not context or not field_query:
        return None
    field = field_query.strip()
    if not field or len(field) > 80:
        return None
    text = context.replace("[source]", "\n").replace("[/source]", "\n")
    for pattern in (_label_value_pattern(field), _label_next_line_pattern(field)):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip()
            if value and len(value) < 500 and value != NOT_FOUND:
                return value
    return None


def _looks_like_heading(line: str) -> bool:
    """Heuristic: detect generic section headings without relying on specific labels."""
    s = line.strip()
    if not s:
        return False
    # Obvious headings with punctuation.
    if s.endswith(":") or s.endswith("-") or (len(s) <= 80 and (":" in s or " - " in s)):
        return True
    # Short, mostly text lines without sentence punctuation also often act as headings,
    # e.g. "Technical Skills", "Professional Summary", etc. (no hardcoded labels).
    if len(s) <= 60 and not any(ch in s for ch in ".!?") and any(c.isalpha() for c in s):
        return True
    return False


def try_section_extract(context: str, key_term: str | None) -> str | None:
    """
    Find a heading line that contains key_term (from the user question), then return that heading and the
    following lines until the next heading or limit. No LLM; works for any document. Returns None if no match.
    """
    if not context or not key_term:
        return None
    key = key_term.strip().lower()
    if not key or len(key) > 60:
        return None
    text = context.replace("[source]", "\n").replace("[/source]", "\n")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if key not in line.lower():
            continue
        if not _looks_like_heading(line):
            continue
        collected = [line.strip()]
        for j in range(i + 1, min(i + 1 + SECTION_EXTRACT_MAX_LINES, len(lines))):
            next_line = lines[j]
            if not next_line.strip():
                collected.append("")
                continue
            if _looks_like_heading(next_line) and len(next_line.strip()) < 100:
                break
            collected.append(next_line.rstrip())
        block = "\n".join(collected).strip()
        if block and len(block) < 8000 and block != NOT_FOUND:
            return block
    return None


def truncate_context_for_extractor(context: str, search_queries: List[str] | None = None) -> str:
    """
    Limit context before sending to extractor LLM (config-driven). If search_queries given,
    take a window starting at the first occurrence of any query term so the LLM sees less, faster.
    """
    if not context:
        return ""
    max_len = EXTRACTOR_MAX_CONTEXT_CHARS
    if len(context) <= max_len:
        return context

    start = 0
    if search_queries:
        context_lower = context.lower()
        for q in search_queries:
            if not (q and q.strip()):
                continue
            for part in q.strip().lower().split():
                if len(part) < 2:
                    continue
                pos = context_lower.find(part)
                if pos != -1 and (start == 0 or pos < start):
                    start = pos
        if start > 0:
            line_start = context.rfind("\n", 0, start)
            start = line_start + 1 if line_start != -1 else start

    end = min(len(context), start + max_len)
    cut = context[start:end]
    if end < len(context):
        last_nl = cut.rfind("\n")
        if last_nl > len(cut) // 2:
            cut = cut[: last_nl + 1]
    return cut


def _search_node(state: AgentState, shared: Dict[str, object]) -> AgentState:
    """Search node: use field extraction or short-query bypass to skip LLM when possible; else expand into 1–2 queries."""
    question = (state.get("question") or "").strip()
    if not question:
        return state

    field_query = _extract_field_query(question)
    if field_query:
        new_state: AgentState = dict(state)
        new_state["search_queries"] = [field_query]
        return new_state

    if _is_short_query(question):
        new_state = dict(state)
        new_state["search_queries"] = [question]
        return new_state

    multi_chain = shared["multi_chain"]
    try:
        raw = multi_chain.invoke({"question": question})
        text = getattr(raw, "content", None) or getattr(raw, "text", None) or str(raw)
        search_queries = [q.strip() for q in text.split("\n") if q.strip()][:2]
        if not search_queries:
            search_queries = [question]
    except Exception:
        search_queries = [question]

    new_state = dict(state)
    new_state["search_queries"] = search_queries
    return new_state


def _retrieval_node(state: AgentState, shared: Dict[str, object]) -> AgentState:
    """Retrieval node: execute vector searches (in parallel) and return [source]-wrapped context."""
    search_queries = state.get("search_queries") or []
    if not search_queries:
        question = (state.get("question") or "").strip()
        if question:
            search_queries = [question]
        else:
            new_state: AgentState = dict(state)
            new_state["context"] = ""
            return new_state

    base_retriever = shared["base_retriever"]
    vectorstore = shared["vectorstore"]

    docs = []
    seen = set()

    def _retrieve_one(q: str):
        try:
            return base_retriever.invoke(q)
        except Exception:
            return []

    # Parallelize retrieval across search queries to reduce latency.
    max_workers = max(1, min(len(search_queries), 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_retrieve_one, search_queries))

    for cur_docs in results:
        for d in cur_docs:
            key = (getattr(d, "page_content", None), tuple(sorted((d.metadata or {}).items())))
            if key not in seen:
                seen.add(key)
                docs.append(d)

    if not docs:
        new_state: AgentState = dict(state)
        new_state["context"] = NOT_FOUND
        return new_state

    if len(search_queries) > 1:
        docs = _expand_window(vectorstore, docs, RETRIEVAL_WINDOW_SIZE)
        if not docs:
            new_state = dict(state)
            new_state["context"] = NOT_FOUND
            return new_state

    context_text = _format_docs(docs, exclude_queries=search_queries)
    if not context_text:
        context_text = NOT_FOUND
    new_state: AgentState = dict(state)
    new_state["context"] = context_text
    return new_state


def _extractor_node(state: AgentState, shared: Dict[str, object]) -> AgentState:
    """Extractor node: strict SYSTEM_RAG to produce final verbatim output."""
    context = (state.get("context") or "").strip()
    question = (state.get("question") or "").strip()

    if not context or context == NOT_FOUND or not question:
        new_state: AgentState = dict(state)
        new_state["answer"] = NOT_FOUND
        return new_state

    context = truncate_context_for_extractor(context, state.get("search_queries"))
    extractor_chain = shared["extractor_chain"]
    try:
        result = extractor_chain.invoke({"context": context, "question": question})
        text = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
    except Exception:
        text = NOT_FOUND

    new_state: AgentState = dict(state)
    new_state["answer"] = text or NOT_FOUND
    return new_state


def build_rag_agent():
    """Compile a LangGraph with Search, Retrieval, and Extractor nodes."""
    shared = _build_shared_components()

    def search_node(state: AgentState) -> AgentState:
        return _search_node(state, shared)

    def retrieval_node(state: AgentState) -> AgentState:
        return _retrieval_node(state, shared)

    def extractor_node(state: AgentState) -> AgentState:
        return _extractor_node(state, shared)

    graph = StateGraph(AgentState)
    graph.add_node("search", search_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("extract", extractor_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", END)

    return graph.compile()


def get_rag_agent():
    # Stateless compile; can be memoized by the caller if desired.
    return build_rag_agent()


def run_search_retrieval(question: str) -> AgentState:
    """Run only Search and Retrieval nodes; returns state with context. Used so we can stream the Extractor separately."""
    shared = _build_shared_components()
    state: AgentState = {"question": (question or "").strip()}
    state = _search_node(state, shared)
    state = _retrieval_node(state, shared)
    return state


def get_extractor_chain():
    """Return the extractor chain for streaming (same as used by the graph)."""
    return _build_shared_components()["extractor_chain"]

