from __future__ import annotations

import logging

from paperless.config import AIConfig
from paperless.models import LLMBackend

logger = logging.getLogger(
    "paperless_ai.llm"
)

def get_llm():

    config = AIConfig()

    match config.llm_backend:

        case LLMBackend.OPENAI_LIKE:

            from llama_index.llms.openai_like import (
                OpenAILike,
            )

            return OpenAILike(

                model=config.llm_model,

                api_key=config.llm_api_key,

                api_base=config.llm_endpoint,

                temperature=0,

                max_tokens=config.llm_context_size,

            )

        case LLMBackend.OLLAMA:

            from llama_index.llms.ollama import (
                Ollama,
            )

            return Ollama(

                model=config.llm_model,

                base_url=(
                    config.llm_endpoint
                    or "http://localhost:11434"
                ),

                request_timeout=300,

                temperature=0,

            )

        case _:

            raise RuntimeError(

                f"Unsupported LLM backend: "

                f"{config.llm_backend}"

            )
        
def llm_ready():

    try:

        get_llm()

        return True

    except Exception as e:

        logger.exception(e)

        return False
    
def complete(
    prompt: str,
):
    """
    Call LLM.

    Input

        Prompt

    Output

        Completion
    """

    llm = get_llm()

    response = llm.complete(
        prompt,
    )

    return response.text

def complete(
    prompt: str,
):
    """
    Call LLM.

    Input

        Prompt

    Output

        Completion
    """

    llm = get_llm()

    response = llm.complete(
        prompt,
    )

    return response.text

def build_context(
    chunks: list[str],
) -> str:
    """
    Build context.

    Input

        [
            chunk1,
            chunk2,
            chunk3
        ]

    Output

        Context String
    """

    if not chunks:

        return ""

    return "\n\n".join(

        chunks,

    )

def build_prompt(
    query: str,
    context: str,
) -> str:
    """
    Build RAG prompt.
    """

    return f"""
{SYSTEM_PROMPT}

==========================
CONTEXT
==========================

{context}

==========================
QUESTION
==========================

{query}

==========================
ANSWER
==========================

"""

def build_prompt_from_chunks(
    query: str,
    chunks: list[dict],
) -> str:
    """
    Build prompt directly
    from reranked chunks.
    """

    context = build_context(

        [

            chunk["text"]

            for chunk in chunks

        ]

    )

    return build_prompt(

        query,

        context,

    )

def print_prompt(
    prompt: str,
):
    """
    Print prompt.
    """

    logger.info("=" * 80)

    logger.info(prompt)

    logger.info("=" * 80)

def ask(
    query: str,
    context: str,
) -> str:
    """
    Ask LLM using query + context.
    """

    prompt = build_prompt(

        query,

        context,

    )

    logger.info("=" * 80)

    logger.info(
        "Sending prompt to LLM..."
    )

    answer = complete(
        prompt,
    )

    logger.info(
        "LLM completed."
    )

    return answer

def ask_from_chunks(
    query: str,
    chunks: list[dict],
) -> str:
    """
    Ask LLM using reranked chunks.
    """

    prompt = build_prompt_from_chunks(

        query,

        chunks,

    )

    logger.info(
        "Generating answer..."
    )

    return complete(
        prompt,
    )

def ask_from_context(
    query: str,
    contexts: list[str],
) -> str:
    """
    Ask LLM using context list.
    """

    context = build_context(

        contexts,

    )

    return ask(

        query,

        context,

    )

def stream_answer(
    prompt: str,
):
    """
    Stream LLM response.
    """

    llm = get_llm()

    for token in llm.stream_complete(
        prompt,
    ):

        yield token.delta

def stream_from_chunks(
    query: str,
    chunks: list[dict],
):
    """
    Stream answer from chunks.
    """

    prompt = build_prompt_from_chunks(

        query,

        chunks,

    )

    yield from stream_answer(
        prompt,
    )

def print_answer(
    answer: str,
):
    """
    Print answer.
    """

    logger.info("=" * 80)

    logger.info(answer)

    logger.info("=" * 80)

def answer_question(
    query: str,
    chunks: list[dict],
) -> dict:
    """
    Complete RAG pipeline.

    Query
        ↓
    Prompt
        ↓
    LLM
        ↓
    Answer
    """

    answer = ask_from_chunks(

        query,

        chunks,

    )

    return {

        "query": query,

        "answer": answer,

        "context_chunks": len(chunks),

    }

def answer_with_sources(
    query: str,
    chunks: list[dict],
) -> dict:
    """
    Return answer together
    with document sources.
    """

    answer = ask_from_chunks(

        query,

        chunks,

    )

    sources = []

    seen = set()

    for chunk in chunks:

        document_id = chunk["document_id"]

        if document_id in seen:

            continue

        seen.add(document_id)

        sources.append(

            {

                "document_id":
                    document_id,

                "title":
                    chunk.get("title"),

                "chunk_index":
                    chunk.get("chunk_index"),

            }

        )

    return {

        "answer": answer,

        "sources": sources,

    }

def safe_answer(
    query: str,
    chunks: list[dict],
):
    """
    Never raise exception.
    """

    try:

        return answer_with_sources(

            query,

            chunks,

        )

    except Exception as e:

        logger.exception(e)

        return {

            "answer":

                "Xin lỗi, hệ thống "

                "không thể tạo câu trả lời.",

            "sources": [],

        }
    
def log_llm_summary(
    query: str,
    chunks: list[dict],
):
    """
    Print summary.
    """

    logger.info("=" * 80)

    logger.info(

        "Query : %s",

        query,

    )

    logger.info(

        "Context Chunks : %d",

        len(chunks),

    )

    logger.info("=" * 80)

__all__ = [

    "get_llm",

    "llm_ready",

    "complete",

    "stream_answer",

    "build_context",

    "build_prompt",

    "build_prompt_from_chunks",

    "ask",

    "ask_from_context",

    "ask_from_chunks",

    "answer_question",

    "answer_with_sources",

    "safe_answer",

    "print_prompt",

    "print_answer",

    "log_llm_summary",

]

