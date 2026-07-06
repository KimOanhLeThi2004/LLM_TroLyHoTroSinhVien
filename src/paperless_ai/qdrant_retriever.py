import os
from typing import Any

import httpx

QDRANT_URL_ENV = "PAPERLESS_AI_QDRANT_URL"
QDRANT_API_KEY_ENV = "PAPERLESS_AI_QDRANT_API_KEY"
QDRANT_COLLECTION_ENV = "PAPERLESS_AI_QDRANT_COLLECTION"
QDRANT_VECTOR_SIZE_ENV = "PAPERLESS_AI_QDRANT_VECTOR_SIZE"


def _get_qdrant_settings() -> dict[str, Any]:
    return {
        "url": os.getenv(QDRANT_URL_ENV, "").strip(),
        "api_key": os.getenv(QDRANT_API_KEY_ENV, "").strip(),
        "collection": os.getenv(QDRANT_COLLECTION_ENV, "").strip(),
        "vector_size": os.getenv(QDRANT_VECTOR_SIZE_ENV, "1536").strip(),
    }


def retrieve_context_from_qdrant(query: str, top_k: int = 5) -> list[str]:
    settings = _get_qdrant_settings()
    if not settings["url"] or not settings["collection"]:
        return []

    url = settings["url"].rstrip("/") + "/collections/" + settings["collection"] + "/points/query"
    headers = {"Content-Type": "application/json"}
    if settings["api_key"]:
        headers["api-key"] = settings["api_key"]

    payload = {
        "query": query,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()

    data = response.json()
    points = data.get("result") or []
    return [str(point.get("payload", {}).get("text", "")) for point in points if point.get("payload", {}).get("text")]
