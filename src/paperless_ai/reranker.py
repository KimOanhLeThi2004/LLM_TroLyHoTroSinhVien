from __future__ import annotations

import logging
from functools import lru_cache

from FlagEmbedding import FlagReranker

logger = logging.getLogger(
    "paperless_ai.reranker"
)


DEFAULT_MODEL = (
    "BAAI/bge-reranker-v2-m3"
)


@lru_cache(maxsize=1)
def get_reranker() -> FlagReranker:
    """
    Load reranker only once.

    The model is cached so every query
    reuses the same instance.
    """

    logger.info(
        "Loading reranker: %s",
        DEFAULT_MODEL,
    )

    model = FlagReranker(
        DEFAULT_MODEL,
        use_fp16=True,
    )

    logger.info(
        "Reranker loaded."
    )

    return model

def reranker_ready() -> bool:
    """
    Check reranker status.
    """

    try:

        get_reranker()

        return True

    except Exception as e:

        logger.exception(e)

        return False
    
def compute_score(
    query: str,
    passage: str,
) -> float:
    """
    Compute relevance score.

    Input:

        Query
            +

        Passage

    Output:

        Float score
    """

    model = get_reranker()

    score = model.compute_score(

        [

            query,

            passage,

        ]

    )

    return float(score)

def rerank_chunks(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved chunks.

    Flow

        Query
            +
        Retrieved Chunk
                ↓
        BGE Reranker
                ↓
        Score
                ↓
        Sort
                ↓
        Top K
    """

    if not chunks:

        return []

    model = get_reranker()

    logger.info(
        "Reranking %d chunks...",
        len(chunks),
    )

    pairs = [

        [

            query,

            chunk["text"],

        ]

        for chunk in chunks

    ]

    scores = model.compute_score(
        pairs,
    )

    for chunk, score in zip(
        chunks,
        scores,
        strict=False,
    ):

        chunk["rerank_score"] = float(
            score,
        )

    chunks.sort(

        key=lambda x: x["rerank_score"],

        reverse=True,

    )

    logger.info(
        "Top %d chunks selected.",
        top_k,
    )

    return chunks[:top_k]

def best_chunk(
    query: str,
    chunks: list[dict],
):
    """
    Return the best chunk.
    """

    results = rerank_chunks(

        query,

        chunks,

        top_k=1,

    )

    if not results:

        return None

    return results[0]

def print_scores(
    chunks: list[dict],
):
    """
    Print reranking scores.
    """

    logger.info("=" * 80)

    for chunk in chunks:

        logger.info(

            "[%.4f] %s",

            chunk.get(
                "rerank_score",
                0,
            ),

            chunk["text"][:120],

        )

from collections import defaultdict


def rerank_documents(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank documents.

    Flow

        Top 20 Chunks
              ↓
        Group by Document
              ↓
      Keep Best Chunk Score
              ↓
        Sort Documents
              ↓
          Top K
    """

    if not chunks:

        return []

    chunks = rerank_chunks(

        query,

        chunks,

        top_k=len(chunks),

    )

    documents = {}

    for chunk in chunks:

        document_id = chunk["document_id"]

        score = chunk["rerank_score"]

        if document_id not in documents:

            documents[document_id] = {

                "document_id": document_id,

                "title": chunk.get("title"),

                "score": score,

                "best_chunk": chunk,

                "chunks": [chunk],

            }

        else:

            documents[document_id]["chunks"].append(
                chunk,
            )

            if score > documents[document_id]["score"]:

                documents[document_id]["score"] = score

                documents[document_id]["best_chunk"] = chunk

    results = sorted(

        documents.values(),

        key=lambda x: x["score"],

        reverse=True,

    )

    return results[:top_k]

def best_document(
    query: str,
    chunks: list[dict],
):
    """
    Return best matched document.
    """

    docs = rerank_documents(

        query,

        chunks,

        top_k=1,

    )

    if not docs:

        return None

    return docs[0]

def print_document_scores(
    documents: list[dict],
):
    """
    Print reranked documents.
    """

    logger.info("=" * 80)

    for document in documents:

        logger.info(

            "[%.4f] %s",

            document["score"],

            document["title"],

        )

def rerank_context(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = 5,
) -> str:
    """
    Build context after reranking.

    Flow

        Top 20 Chunks
                ↓
           Reranker
                ↓
           Top 5 Chunks
                ↓
      Remove Duplicate
                ↓
         Build Context
    """

    if not chunks:

        return ""

    reranked = rerank_chunks(

        query,

        chunks,

        top_k=top_k,

    )

    context = []

    seen = set()

    for chunk in reranked:

        text = chunk["text"].strip()

        if text in seen:

            continue

        seen.add(text)

        context.append(text)

    return "\n\n".join(context)

def rerank_context_list(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = 5,
) -> list[str]:
    """
    Return context list.
    """

    reranked = rerank_chunks(

        query,

        chunks,

        top_k=top_k,

    )

    return [

        chunk["text"]

        for chunk in reranked

    ]

def print_top_chunks(
    chunks: list[dict],
):
    """
    Print reranked chunks.
    """

    logger.info("=" * 80)

    for index, chunk in enumerate(chunks, start=1):

        logger.info(

            "Rank %d",

            index,

        )

        logger.info(

            "Score : %.4f",

            chunk["rerank_score"],

        )

        logger.info(

            "Document : %s",

            chunk.get("title"),

        )

        logger.info(

            "%s",

            chunk["text"][:200],

        )

        logger.info("-" * 80)

def batch_rerank(
    query: str,
    chunks: list[dict],
    *,
    batch_size: int = 32,
) -> list[dict]:
    """
    Batch reranking.

    Useful when the number of chunks
    is very large.
    """

    if not chunks:

        return []

    model = get_reranker()

    logger.info(

        "Batch reranking %d chunks...",

        len(chunks),

    )

    pairs = [

        [

            query,

            chunk["text"],

        ]

        for chunk in chunks

    ]

    scores = []

    for start in range(

        0,

        len(pairs),

        batch_size,

    ):

        end = start + batch_size

        batch = pairs[start:end]

        batch_scores = model.compute_score(

            batch,

        )

        scores.extend(batch_scores)

    for chunk, score in zip(

        chunks,

        scores,

        strict=False,

    ):

        chunk["rerank_score"] = float(score)

    chunks.sort(

        key=lambda x: x["rerank_score"],

        reverse=True,

    )

    return chunks

def rerank_with_threshold(
    query: str,
    chunks: list[dict],
    *,
    threshold: float = 0.5,
):
    """
    Keep only chunks
    above threshold.
    """

    chunks = batch_rerank(

        query,

        chunks,

    )

    return [

        chunk

        for chunk in chunks

        if chunk["rerank_score"] >= threshold

    ]

def safe_rerank(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = 5,
):
    """
    Safe reranking.

    Never raises exception.
    """

    try:

        return rerank_chunks(

            query,

            chunks,

            top_k=top_k,

        )

    except Exception as e:

        logger.exception(e)

        return chunks[:top_k]
    
def log_rerank_summary(
    chunks: list[dict],
):
    """
    Print reranking summary.
    """

    logger.info("=" * 80)

    logger.info(

        "Chunks : %d",

        len(chunks),

    )

    if chunks:

        logger.info(

            "Best score : %.4f",

            chunks[0]["rerank_score"],

        )

        logger.info(

            "Worst score : %.4f",

            chunks[-1]["rerank_score"],

        )

    logger.info("=" * 80)

__all__ = [

    "get_reranker",

    "reranker_ready",

    "compute_score",

    "rerank_chunks",

    "rerank_documents",

    "best_chunk",

    "best_document",

    "rerank_context",

    "rerank_context_list",

    "batch_rerank",

    "rerank_with_threshold",

    "safe_rerank",

    "print_scores",

    "print_document_scores",

    "print_top_chunks",

    "log_rerank_summary",

]