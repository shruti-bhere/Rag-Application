"""
Streaming handler: runs Search -> Retrieval, then streams the Extractor LLM token-by-token
so the user sees output immediately instead of a long "Thinking…" wait.
"""
import asyncio
import json
from typing import AsyncGenerator

from backend.graph import (
    NOT_FOUND,
    get_extractor_chain,
    get_field_query,
    run_search_retrieval,
    truncate_context_for_extractor,
    try_fast_extract,
    try_section_extract,
)


async def stream_agent_response(agent, question: str) -> AsyncGenerator[str, None]:
    """Run search+retrieval; if we can extract with a pattern (Label: value), return instantly. Else stream the extractor LLM."""
    try:
        # Run search + retrieval in thread (sync) so we don't block the event loop.
        state = await asyncio.to_thread(run_search_retrieval, question)
        context = (state or {}).get("context") or ""
        question_str = (state or {}).get("question") or question

        if not context or context == NOT_FOUND:
            yield f"data: {json.dumps({'type': 'token', 'content': NOT_FOUND})}\n\n"
            yield "data: [DONE]\n\n"
            return

        field = get_field_query(question_str)
        if not field and (state or {}).get("search_queries"):
            sq = state["search_queries"]
            if len(sq) == 1 and len(sq[0].split()) <= 1:
                field = sq[0]
        fast_value = try_fast_extract(context, field)
        if fast_value:
            yield f"data: {json.dumps({'type': 'token', 'content': fast_value})}\n\n"
            yield "data: [DONE]\n\n"
            return

        sq = (state or {}).get("search_queries") or [question_str]
        parts = (sq[0].strip().split() if sq and sq[0].strip() else question_str.strip().split()) or []
        section_value = None
        for key_term in (parts[-1], parts[0]) if len(parts) > 1 else (parts[0],) if parts else ():
            section_value = try_section_extract(context, key_term)
            if section_value:
                break
        if section_value:
            yield f"data: {json.dumps({'type': 'token', 'content': section_value})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context = truncate_context_for_extractor(context, search_queries=sq)
        chain = get_extractor_chain()
        async for chunk in chain.astream({"context": context, "question": question_str}):
            content = None
            if isinstance(chunk, str):
                content = chunk
            elif hasattr(chunk, "content"):
                content = getattr(chunk, "content", None)
                if content is not None and not isinstance(content, str):
                    content = str(content)
            elif isinstance(chunk, dict):
                for v in chunk.values():
                    if hasattr(v, "content"):
                        c = getattr(v, "content", None)
                        content = c if isinstance(c, str) else (str(c) if c else None)
                        break
                    if isinstance(v, str):
                        content = v
                        break
            if content:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    yield "data: [DONE]\n\n"

