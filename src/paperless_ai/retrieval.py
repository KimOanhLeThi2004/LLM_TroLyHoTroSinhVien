from __future__ import annotations

import logging
from typing import Iterable

from django.conf import settings

from documents.models import Document

from paperless_ai.embedding import (
    build_document_text,
    embed_query,
)

logger = logging.getLogger(
    "paperless_ai.retrieval"
)

def normalize_document_ids(
    document_ids: Iterable[int | str] | None,
) -> set[str] | None:

    if document_ids is None:

        return None

    return {

        str(doc_id)

        for doc_id in document_ids

    }

def normalize_query(
    query: str,
) -> str:
    """
    Normalize query before embedding.
    """

    if not query:

        return ""

    query = query.replace(
        "\r",
        "\n",
    )

    query = query.strip()

    return query

def build_query(
    query: str,
):
    """
    Build embedding vector
    for semantic search.
    """

    query = normalize_query(
        query,
    )

    return embed_query(
        query,
    )


def semantic_search(
    query: str,
    *,
    top_k: int = 20,
):
    """
    Semantic Search

        Query
            ↓
        Embedding
            ↓
        Vector Search
            ↓
        Top K Chunks
    """

    logger.info("=" * 80)

    logger.info(
        "Semantic Search"
    )

    query = normalize_query(
        query,
    )

    if not query:

        return []

    from paperless_ai.qdrant_store import (
        semantic_search_qdrant,
    )

    results = semantic_search_qdrant(

        query=query,

        top_k=top_k,

    )

    logger.info(
        "Retrieved %d chunks",
        len(results),
    )

    return results

def search_document(
    query: str,
    *,
    top_k: int = 20,
):

    return semantic_search(

        query,

        top_k=top_k,

    )

def build_context(
    chunks: list[dict],
    *,
    max_chunks: int = 10,
) -> str:
    """
    Build context for LLM.

    Semantic Search
            ↓
      Top K Chunks
            ↓
      Remove Duplicate
            ↓
       Merge Context
    """

    if not chunks:

        return ""

    context = []

    seen = set()

    for chunk in chunks:

        text = chunk["text"].strip()

        if text in seen:

            continue

        seen.add(text)

        context.append(

            text,

        )

        if len(context) >= max_chunks:

            break

    return "\n\n".join(
        context,
    )

from collections import defaultdict


def group_chunks_by_document(
    chunks: list[dict],
):
    """
    Group retrieved chunks
    by document.
    """

    grouped = defaultdict(list)

    for chunk in chunks:

        grouped[
            chunk["document_id"]
        ].append(
            chunk,
        )

    return grouped

def unique_documents(
    chunks: list[dict],
):
    """
    Return unique document ids.
    """

    docs = []

    seen = set()

    for chunk in chunks:

        document_id = chunk[
            "document_id"
        ]

        if document_id in seen:

            continue

        seen.add(
            document_id,
        )

        docs.append(
            document_id,
        )

    return docs

from paperless_ai.reranker import (
    rerank_chunks,
)


def retrieve(
    query: str,
    *,
    top_k: int = 20,
):
    """
    Retrieve semantic search result.

    Return

    [
        {
            ...
        }
    ]
    """

    logger.info("=" * 80)

    logger.info(
        "Retrieving..."
    )

    chunks = search_document(

        query,

        top_k=top_k,

    )

    logger.info(

        "Retrieved %d chunks.",

        len(chunks),

    )

    return chunks

def retrieve_with_rerank(
    query: str,
    *,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
):
    """
    Retrieval

        Semantic Search

            ↓

        Top20

            ↓

        Reranker

            ↓

        Top5
    """

    chunks = retrieve(

        query,

        top_k=retrieve_top_k,

    )

    if not chunks:

        return []

    logger.info(

        "Reranking..."

    )

    chunks = rerank_chunks(

        query,

        chunks,

        top_k=rerank_top_k,

    )

    logger.info(

        "Reranked %d chunks.",

        len(chunks),

    )

    return chunks

def retrieve_context(
    query: str,
    *,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
):
    """
    Retrieve context
    for LLM.
    """

    chunks = retrieve_with_rerank(

        query,

        retrieve_top_k=retrieve_top_k,

        rerank_top_k=rerank_top_k,

    )

    context = build_context(

        chunks,

        max_chunks=rerank_top_k,

    )

    return context

def retrieve_by_document(
    query: str,
    document_id: str,
    *,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
):
    """
    Search inside one document only.
    """

    chunks = retrieve(

        query,

        top_k=retrieve_top_k,

    )

    chunks = [

        chunk

        for chunk in chunks

        if chunk["document_id"] == str(document_id)

    ]

    if not chunks:

        return []

    return rerank_chunks(

        query,

        chunks,

        top_k=rerank_top_k,

    )

def retrieve_by_documents(
    query: str,
    document_ids: Iterable[int | str],
    *,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
):
    """
    Search inside multiple documents.
    """

    allowed = normalize_document_ids(
        document_ids,
    )

    chunks = retrieve(

        query,

        top_k=retrieve_top_k,

    )

    chunks = [

        chunk

        for chunk in chunks

        if chunk["document_id"] in allowed

    ]

    if not chunks:

        return []

    return rerank_chunks(

        query,

        chunks,

        top_k=rerank_top_k,

    )

def retrieve_by_score(
    query: str,
    *,
    min_score: float = 0.5,
    retrieve_top_k: int = 20,
):
    """
    Filter by semantic similarity score.
    """

    chunks = retrieve(

        query,

        top_k=retrieve_top_k,

    )

    return [

        chunk

        for chunk in chunks

        if chunk["score"] >= min_score

    ]

def retrieve_by_rerank_score(
    query: str,
    *,
    threshold: float = 0.5,
    retrieve_top_k: int = 20,
):
    """
    Filter by reranker score.
    """

    chunks = retrieve_with_rerank(

        query,

        retrieve_top_k=retrieve_top_k,

        rerank_top_k=retrieve_top_k,

    )

    return [

        chunk

        for chunk in chunks

        if chunk["rerank_score"] >= threshold

    ]

def retrieve_best_chunk(
    query: str,
):
    """
    Return the best chunk.
    """

    chunks = retrieve_with_rerank(

        query,

        retrieve_top_k=20,

        rerank_top_k=1,

    )

    if not chunks:

        return None

    return chunks[0]

def retrieve_context_chunks(
    query: str,
    *,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
):
    """
    Return reranked chunks.

    Used for LLM.
    """

    return retrieve_with_rerank(

        query,

        retrieve_top_k=retrieve_top_k,

        rerank_top_k=rerank_top_k,

    )

def search(
    query: str,
    *,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 5,
):
    """
    Complete Retrieval Pipeline.

        Query
            ↓
      Semantic Search
            ↓
          Top20
            ↓
        Reranker
            ↓
          Top5
            ↓
        Build Context
    """

    chunks = retrieve_with_rerank(

        query,

        retrieve_top_k=retrieve_top_k,

        rerank_top_k=rerank_top_k,

    )

    context = build_context(

        chunks,

        max_chunks=rerank_top_k,

    )

    return {

        "query": query,

        "chunks": chunks,

        "context": context,

    }

from paperless_ai.llm import (
    answer_with_sources,
)


def ask(
    query: str,
):
    """
    Complete RAG Pipeline.

        Query
            ↓
      Semantic Search
            ↓
        Reranker
            ↓
      Context Builder
            ↓
            LLM
    """

    chunks = retrieve_with_rerank(

        query,

        retrieve_top_k=20,

        rerank_top_k=5,

    )

    return answer_with_sources(

        query,

        chunks,

    )

def retrieval_ready():
    """
    Check retrieval pipeline.
    """

    try:

        build_query(
            "test",
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False
    
def log_summary(
    query: str,
    chunks: list[dict],
):
    """
    Print retrieval summary.
    """

    logger.info("=" * 80)

    logger.info(

        "Query : %s",

        query,

    )

    logger.info(

        "Retrieved : %d",

        len(chunks),

    )

    if chunks:

        logger.info(

            "Best Score : %.4f",

            chunks[0].get(

                "rerank_score",

                chunks[0]["score"],

            ),

        )

    logger.info("=" * 80)

__all__ = [

    "build_query",

    "normalize_query",

    "semantic_search",

    "search_document",

    "retrieve",

    "retrieve_with_rerank",

    "retrieve_context",

    "retrieve_context_chunks",

    "retrieve_by_document",

    "retrieve_by_documents",

    "retrieve_by_score",

    "retrieve_by_rerank_score",

    "retrieve_best_chunk",

    "build_context",

    "group_chunks_by_document",

    "unique_documents",

    "search",

    "ask",

    "retrieval_ready",

]