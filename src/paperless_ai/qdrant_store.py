from __future__ import annotations

import logging

from django.conf import settings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import VectorParams
from qdrant_client.http.models import PointStruct

from paperless_ai.embedding import get_embedding_dim

logger = logging.getLogger(
    "paperless_ai.qdrant"
)

COLLECTION_NAME = "paperless_documents"

from urllib.parse import urlparse


def get_qdrant_client() -> QdrantClient:
    """
    Return configured Qdrant client.
    """

    parsed = urlparse(settings.QDRANT_URL)

    return QdrantClient(
        host=parsed.hostname,
        port=parsed.port or 443,
        https=(parsed.scheme == "https"),
        api_key=getattr(
            settings,
            "QDRANT_API_KEY",
            None,
        ),
        check_compatibility=False,
        timeout=300,
    )

def create_collection():

    client = get_qdrant_client()

    collections = client.get_collections()

    names = {

        c.name

        for c in collections.collections

    }

    if COLLECTION_NAME in names:

        return

    dimension = get_embedding_dim()

    logger.info(

        "Create collection : %s",

        COLLECTION_NAME,

    )

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=dimension,

            distance=Distance.COSINE,

        ),

    )

def collection_exists():

    client = get_qdrant_client()

    collections = client.get_collections()

    return COLLECTION_NAME in {

        c.name

        for c in collections.collections

    }

import uuid


def upsert_chunks(
    embedded_chunks,
):
    """
    Upsert chunks into Qdrant.

    embedded_chunks =

    [
        {
            "chunk": Chunk,
            "vector": [...]
        }
    ]
    """

    if not embedded_chunks:

        return

    create_collection()

    client = get_qdrant_client()

    points = []

    for item in embedded_chunks:

        chunk = item["chunk"]

        vector = item["vector"]

        payload = {

            "document_id": chunk.document_id,

            "chunk_id": chunk.chunk_id,

            "chunk_index": chunk.chunk_index,

            "text": chunk.text,

            "keywords": getattr(chunk, "keywords", []),

            "topic": getattr(chunk, "topic", ""),

            "layout": getattr(chunk, "layout", {}),
        }

        points.append(

            PointStruct(

                id=str(uuid.uuid4()),

                vector=vector,

                payload=payload,

            )

        )

    logger.info(

        "Upsert %d chunks.",

        len(points),

    )

    BATCH_SIZE = 100

    for i in range(0, len(points), BATCH_SIZE):

        batch = points[i:i + BATCH_SIZE]

        logger.info(

            "Upsert batch %d-%d/%d",

            i + 1,

            min(i + BATCH_SIZE, len(points)),

            len(points),

        )

        client.upsert(

            collection_name=COLLECTION_NAME,

            wait=True,

            points=batch,

        )

def upsert_document(
    chunks,
):
    """
    Embed + Upsert document.
    """

    if not chunks:

        return

    from paperless_ai.embedding import (
        embed_chunks,
    )

    document_id = chunks[0].document_id

    if document_id is not None:

        delete_document(
            document_id,
        )

    embedded = embed_chunks(
        chunks,
    )

    upsert_chunks(
        embedded,
    )

def count_vectors():

    client = get_qdrant_client()

    result = client.count(

        collection_name=COLLECTION_NAME,

        exact=True,

    )

    return result.count

def collection_info():

    client = get_qdrant_client()

    return client.get_collection(

        COLLECTION_NAME,

    )

from qdrant_client.http.models import Filter


def search(
    query_vector,
    *,
    top_k: int = 20,
    query_filter: Filter | None = None,
):
    """
    Semantic Search

        Query Vector
              ↓
          Qdrant
              ↓
          Top K
    """

    create_collection()

    client = get_qdrant_client()

    logger.info(
        "Searching top %d chunks...",
        top_k,
    )

    results = client.search(

        collection_name=COLLECTION_NAME,

        query_vector=query_vector,

        query_filter=query_filter,

        limit=top_k,

        with_payload=True,

        with_vectors=False,

    )

    return results

from qdrant_client.http.models import (
    FieldCondition,
    MatchValue,
    Filter,
)


def search_by_document(
    query_vector,
    document_id: str,
    *,
    top_k: int = 20,
):
    """
    Search only inside one document.
    """

    query_filter = Filter(

        must=[

            FieldCondition(

                key="document_id",

                match=MatchValue(
                    value=document_id,
                ),

            )

        ]

    )

    return search(

        query_vector,

        top_k=top_k,

        query_filter=query_filter,

    )

from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
)


def delete_document(
    document_id: str,
):
    """
    Delete all vectors
    of one document.
    """

    if not collection_exists():

        return

    client = get_qdrant_client()

    logger.info(

        "Delete document %s",

        document_id,

    )

    client.delete(

        collection_name=COLLECTION_NAME,

        points_selector=Filter(

            must=[

                FieldCondition(

                    key="document_id",

                    match=MatchValue(

                        value=document_id,

                    ),

                )

            ]

        ),

    )

def delete_collection():
    """
    Delete Qdrant collection.
    """

    client = get_qdrant_client()

    if collection_exists():

        logger.info(

            "Delete collection %s",

            COLLECTION_NAME,

        )

        client.delete_collection(

            COLLECTION_NAME,

        )

def recreate_collection():
    """
    Delete and recreate collection.
    """

    delete_collection()

    create_collection()

def scroll_document_chunks(
    document_id: str,
):
    """
    Return all chunks of one document.
    """

    from qdrant_client.http.models import (
        Filter,
        FieldCondition,
        MatchValue,
    )

    client = get_qdrant_client()

    points, _ = client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=Filter(

            must=[

                FieldCondition(

                    key="document_id",

                    match=MatchValue(
                        value=document_id,
                    ),

                )

            ]

        ),

        with_payload=True,

        with_vectors=False,

        limit=10000,

    )

    return points

def get_document_chunk_count(
    document_id: str,
) -> int:
    """
    Return number of chunks
    of one document.
    """

    points = scroll_document_chunks(
        document_id,
    )

    return len(points)

def health_check():
    """
    Check Qdrant connection.
    """

    try:

        client = get_qdrant_client()

        client.get_collections()

        logger.info(
            "Qdrant connected."
        )

        return True

    except Exception as e:

        logger.exception(e)

        return False
    
def collection_stats():
    """
    Return collection statistics.
    """

    client = get_qdrant_client()

    info = client.get_collection(
        COLLECTION_NAME,
    )

    return {

        "collection":
            COLLECTION_NAME,

        "vectors":
            info.points_count,

        "segments":
            info.segments_count,

        "status":
            str(info.status),

        "dimension":
            info.config.params.vectors.size,

        "distance":
            str(
                info.config.params.vectors.distance
            ),

    }

def print_collection_stats():
    """
    Print collection statistics.
    """

    stats = collection_stats()

    logger.info("=" * 80)

    logger.info(
        "Collection : %s",
        stats["collection"],
    )

    logger.info(
        "Vectors : %s",
        stats["vectors"],
    )

    logger.info(
        "Segments : %s",
        stats["segments"],
    )

    logger.info(
        "Dimension : %s",
        stats["dimension"],
    )

    logger.info(
        "Distance : %s",
        stats["distance"],
    )

    logger.info(
        "Status : %s",
        stats["status"],
    )

def reset_collection():
    """
    Reset collection.
    """

    logger.info(
        "Reset collection..."
    )

    delete_collection()

    create_collection()

    logger.info(
        "Collection ready."
    )

from paperless_ai.embedding import embed_query


def semantic_search_qdrant(
    query: str,
    *,
    top_k: int = 20,
):
    """
    Semantic search using Qdrant.

    Returns

    [
        {
            "document_id": ...,
            "chunk_id": ...,
            "chunk_index": ...,
            "score": ...,
            "text": ...,
            "metadata": {...},
        }
    ]
    """

    vector = embed_query(query)

    results = search(

        vector,

        top_k=top_k,

    )

    chunks = []

    for point in results:

        payload = point.payload or {}

        chunks.append(

            {

                "document_id":
                    payload.get("document_id"),

                "chunk_id":
                    payload.get("chunk_id"),

                "chunk_index":
                    payload.get("chunk_index"),

                "score":
                    point.score,

                "text":
                    payload.get("text", ""),

                "metadata":
                    payload,

            }

        )

    return chunks
