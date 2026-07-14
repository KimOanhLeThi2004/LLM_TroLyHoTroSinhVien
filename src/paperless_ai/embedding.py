from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from pathlib import Path

from django.conf import settings

from llama_index.core.base.embeddings.base import BaseEmbedding

from documents.models import Document
from paperless.config import AIConfig
from paperless.models import LLMEmbeddingBackend

from paperless.network import (
    PinnedHostAsyncHTTPTransport,
    PinnedHostHTTPTransport,
    create_pinned_async_httpx_client,
    create_pinned_httpx_client,
    validate_outbound_http_url,
)

OCR_LEADER_REGEX = re.compile(r"[._\-\u00b7]{4,}")

HORIZONTAL_WHITESPACE_REGEX = re.compile(
    r"[ \t\u00a0]+"
)

def get_embedding_model() -> "BaseEmbedding":

    config = AIConfig()

    match config.llm_embedding_backend:

        case LLMEmbeddingBackend.OPENAI_LIKE:

            from llama_index.embeddings.openai_like import (
                OpenAILikeEmbedding,
            )

            endpoint = (
                config.llm_embedding_endpoint
                or config.llm_endpoint
                or None
            )

            http_client = None
            async_http_client = None

            if endpoint:

                http_client = create_pinned_httpx_client(
                    endpoint,
                    allow_internal=config.llm_allow_internal_endpoints,
                )

                async_http_client = (
                    create_pinned_async_httpx_client(
                        endpoint,
                        allow_internal=config.llm_allow_internal_endpoints,
                    )
                )

            return OpenAILikeEmbedding(

                model_name=(
                    config.llm_embedding_model
                    or "text-embedding-3-small"
                ),

                api_key=config.llm_api_key,

                api_base=endpoint,

                http_client=http_client,

                async_http_client=async_http_client,

            )

        case LLMEmbeddingBackend.HUGGINGFACE:

            from llama_index.embeddings.huggingface import (
                HuggingFaceEmbedding,
            )

            return HuggingFaceEmbedding(

                model_name=(
                    config.llm_embedding_model
                    or "sentence-transformers/all-MiniLM-L6-v2"
                ),

                cache_folder=str(
                    settings.DATA_DIR / "hf_cache"
                ),

            )

        case LLMEmbeddingBackend.OLLAMA:

            from llama_index.embeddings.ollama import (
                OllamaEmbedding,
            )

            from ollama import AsyncClient
            from ollama import Client

            endpoint = (
                config.llm_embedding_endpoint
                or config.llm_endpoint
                or "http://localhost:11434"
            )

            validate_outbound_http_url(
                endpoint,
                allow_internal=config.llm_allow_internal_endpoints,
            )

            embedding = OllamaEmbedding(

                model_name=(
                    config.llm_embedding_model
                    or "embeddinggemma"
                ),

                base_url=endpoint,

                ollama_additional_kwargs={
                    "num_ctx": config.llm_context_size,
                },

            )

            embedding._client = Client(

                host=endpoint,

                transport=PinnedHostHTTPTransport(
                    allow_internal=config.llm_allow_internal_endpoints,
                ),

            )

            embedding._async_client = AsyncClient(

                host=endpoint,

                transport=PinnedHostAsyncHTTPTransport(
                    allow_internal=config.llm_allow_internal_endpoints,
                ),

            )

            return embedding

        case _:

            raise ValueError(

                f"Unsupported embedding backend: "

                f"{config.llm_embedding_backend}"

            )

def get_embedding_dim() -> int:
    """
    Return embedding dimension.

    The dimension is cached in meta.json.
    """

    config = AIConfig()

    default_model = {

        LLMEmbeddingBackend.OPENAI_LIKE:
            "text-embedding-3-small",

        LLMEmbeddingBackend.HUGGINGFACE:
            "sentence-transformers/all-MiniLM-L6-v2",

        LLMEmbeddingBackend.OLLAMA:
            "embeddinggemma",

    }.get(

        config.llm_embedding_backend,

        "sentence-transformers/all-MiniLM-L6-v2",

    )

    model = (
        config.llm_embedding_model
        or default_model
    )

    meta_path: Path = (
        settings.LLM_INDEX_DIR
        / "meta.json"
    )

    meta_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if meta_path.exists():

        with meta_path.open() as f:

            meta = json.load(f)

        if meta.get("embedding_model") != model:

            raise RuntimeError(

                f"Embedding model changed "

                f"from {meta.get('embedding_model')} "

                f"to {model}. "

                f"You must rebuild the index."

            )

        return meta["dim"]

    embedding_model = get_embedding_model()

    vector = embedding_model.get_text_embedding(
        "test"
    )

    dim = len(vector)

    with meta_path.open("w") as f:

        json.dump(

            {

                "embedding_model": model,

                "dim": dim,

            },

            f,

        )

    return dim

def _normalize_embedding_text(
    text: str,
) -> str:
    """
    Normalize OCR text before chunking.
    """

    text = OCR_LEADER_REGEX.sub(
        " ",
        text,
    )

    text = HORIZONTAL_WHITESPACE_REGEX.sub(
        " ",
        text,
    )

    text = re.sub(
        r"\n+",
        "\n",
        text,
    )

    return text.strip()


def _normalize_llm_index_text(
    text: str,
) -> str:
    """
    Backwards-compatible name for OCR text normalization.
    """

    return _normalize_embedding_text(
        text,
    )


def _format_metadata_value(
    value,
) -> str | None:
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def build_llm_index_text(
    doc: Document,
) -> str:
    """
    Build the legacy plain-text representation used by older callers/tests.

    The vector index now embeds OCR content only via build_document_text(); this
    helper remains available for code paths that need a readable document blob.
    """

    from documents.models import Note

    tags = [
        tag.name
        for tag in doc.tags.all()
    ]

    notes = [
        note.note
        for note in Note.objects.filter(document=doc)
    ]

    custom_fields = []

    if hasattr(doc, "custom_fields"):
        for custom_field in doc.custom_fields.all():
            custom_fields.append(
                "Custom Field - "
                f"{custom_field.field.name}: {custom_field.value}"
            )

    parts = [
        f"Title: {doc.title}",
        f"Filename: {doc.filename}",
        f"Created: {_format_metadata_value(doc.created)}",
        f"Added: {_format_metadata_value(doc.added)}",
        f"Modified: {_format_metadata_value(doc.modified)}",
        f"Tags: {', '.join(tags)}",
        "Document Type: "
        f"{doc.document_type.name if doc.document_type else ''}",
        "Correspondent: "
        f"{doc.correspondent.name if doc.correspondent else ''}",
        f"Notes: {','.join(notes)}",
    ]

    if custom_fields:
        parts.append(
            "\n".join(custom_fields),
        )

    content = _normalize_llm_index_text(
        doc.content or "",
    )

    parts.append(
        f"Content:\n\n{content}",
    )

    return "\n".join(parts)

def build_document_text(
    doc: Document,
) -> str:
    """
    Return RAW OCR text.

    Metadata is stored separately
    and is NOT embedded.
    """

    return _normalize_embedding_text(

        doc.content or ""

    )

def build_document_metadata(
    doc: Document,
) -> dict:
    """
    Metadata stored as payload in Qdrant.
    Metadata is NOT embedded.
    """

    return {

        "document_id": str(doc.id),

        "title": doc.title,

        "filename": doc.filename,

        "document_type": (
            doc.document_type.name
            if doc.document_type
            else None
        ),

        "tags": [
            tag.name
            for tag in doc.tags.all()
        ],

        "correspondent": (
            doc.correspondent.name
            if doc.correspondent
            else None
        ),

        "storage_path": (
            doc.storage_path.name
            if doc.storage_path
            else None
        ),

        "archive_serial_number":
            doc.archive_serial_number,

        "created": (
            doc.created.isoformat()
            if doc.created
            else None
        ),

        "added": (
            doc.added.isoformat()
            if doc.added
            else None
        ),

        "modified":
            doc.modified.isoformat(),

    }

from paperless_ai.chunking import Chunk
def embed_chunk(
    chunk: Chunk,
    embedding_model=None,
) -> list[float]:
    """
    Embed a single chunk.
    """

    if embedding_model is None:

        embedding_model = get_embedding_model()

    vector = embedding_model.get_text_embedding(
        chunk.text
    )

    return vector

def embed_chunks(
    chunks: list[Chunk],
    embedding_model=None,
):
    """
    Embed all chunks.

    Returns

    [
        {
            "chunk": chunk,
            "vector": [...]
        }
    ]
    """

    if embedding_model is None:

        embedding_model = get_embedding_model()

    results = []

    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):

        vector = embedding_model.get_text_embedding(
            chunk.text
        )

        results.append(

            {

                "chunk": chunk,

                "vector": vector,

            }

        )

    return results

def embed_query(
    query: str,
):
    """
    Embed search query.
    """

    embedding_model = get_embedding_model()

    return embedding_model.get_text_embedding(
        query
    )
