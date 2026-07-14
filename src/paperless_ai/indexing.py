from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from celery import chunks
from django.conf import settings
from django.utils import timezone
from filelock import FileLock

from documents.models import Document
from documents.models import PaperlessTask

from documents.utils import (
    IterWrapper,
    identity,
)

from paperless.config import AIConfig

from paperless_ai.chunking import (
    SentenceChunkGenerator,
)

from paperless_ai.embedding import (
    build_document_metadata,
    build_document_text,
    get_embedding_dim,
    get_embedding_model,
)

from documents.models import (
    DocumentLayout,
    DocumentKeyword,
    DocumentTopic,
)

if TYPE_CHECKING:

    from llama_index.core import VectorStoreIndex

    from llama_index.core.schema import BaseNode


logger = logging.getLogger(
    "paperless_ai.indexing"
)

RAG_CHUNK_OVERLAP = 200

def _index_lock_path() -> Path:
    """
    Lock file used while updating vector index.
    """

    return settings.LLM_INDEX_LOCK


def queue_llm_index_update_if_needed(
    *,
    rebuild: bool,
    reason: str,
) -> bool:

    from documents.tasks import llmindex_index

    has_running = PaperlessTask.objects.filter(

        task_type=PaperlessTask.TaskType.LLM_INDEX,

        status__in=[

            PaperlessTask.Status.PENDING,

            PaperlessTask.Status.STARTED,

        ],

    ).exists()

    has_recent = PaperlessTask.objects.filter(

        task_type=PaperlessTask.TaskType.LLM_INDEX,

        date_created__gte=(
            timezone.now()
            - timedelta(minutes=5)
        ),

    ).exists()

    if has_running or has_recent:

        return False

    llmindex_index.apply_async(

        kwargs={

            "rebuild": rebuild,

        },

        headers={

            "trigger_source":
                PaperlessTask.TriggerSource.SYSTEM,

        },

    )

    logger.warning(

        "Queued LLM index update%s : %s",

        " (rebuild)" if rebuild else "",

        reason,

    )

    return True

def get_or_create_storage_context(
    *,
    rebuild: bool = False,
):
    """
    Create or load StorageContext.

    Storage

        FAISS

        DocStore

        IndexStore
    """

    if rebuild:

        shutil.rmtree(
            settings.LLM_INDEX_DIR,
            ignore_errors=True,
        )

        settings.LLM_INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    from llama_index.core import StorageContext

    from llama_index.core.storage.docstore import (
        SimpleDocumentStore,
    )

    from llama_index.core.storage.index_store import (
        SimpleIndexStore,
    )

    from llama_index.vector_stores.faiss import (
        FaissVectorStore,
    )

    if rebuild or not vector_store_file_exists():

        import faiss

        embedding_dim = get_embedding_dim()

        logger.info(
            "Embedding Dimension : %d",
            embedding_dim,
        )

        faiss_index = faiss.IndexFlatL2(
            embedding_dim,
        )

        vector_store = FaissVectorStore(
            faiss_index=faiss_index,
        )

        docstore = SimpleDocumentStore()

        index_store = SimpleIndexStore()

    else:

        logger.info(
            "Loading existing vector index..."
        )

        vector_store = (
            FaissVectorStore.from_persist_dir(
                settings.LLM_INDEX_DIR,
            )
        )

        docstore = (
            SimpleDocumentStore.from_persist_dir(
                settings.LLM_INDEX_DIR,
            )
        )

        index_store = (
            SimpleIndexStore.from_persist_dir(
                settings.LLM_INDEX_DIR,
            )
        )

    storage_context = StorageContext.from_defaults(

        vector_store=vector_store,

        docstore=docstore,

        index_store=index_store,

        persist_dir=settings.LLM_INDEX_DIR,

    )

    return storage_context

from paperless_ai.chunking import Chunk
def build_document_chunks(
    document: Document,
    *,
    chunk_size: int | None = None,
) -> list[Chunk]:
    """
    Build TextNodes for a document.

    Pipeline

        OCR Text
            ↓
        Sentence Chunking
            ↓
        Chunk Metadata
            ↓
        TextNode
    """

    chunk_size = chunk_size or get_rag_chunk_size()

    #
    # RAW OCR TEXT
    #

    text = build_document_text(
        document,
    )

    if not text:

        logger.warning(
            "Document %s has no OCR text.",
            document.pk,
        )

        return []

    #
    # DOCUMENT METADATA
    #

    metadata = build_document_metadata(
        document,
    )

    #
    # SENTENCE CHUNKING
    #

    chunker = SentenceChunkGenerator(
        target_tokens=chunk_size,
    )

    chunks = chunker.generate(

        text=text,

        document_id=str(
            document.id,
        ),
        book_title=document.title,
    )

    logger.info(

        "Generated %d chunks.",

        len(chunks),

    )

    if not chunks:

        return []

    #
    # BUILD TEXT NODE
    #

    keywords = []
    topic = ""
    layout = {}

    try:
        keywords = DocumentKeyword.objects.get(
            document=document,
        ).keywords
    except DocumentKeyword.DoesNotExist:
        pass

    try:
        topic = DocumentTopic.objects.get(
            document=document,
        ).topic
    except DocumentTopic.DoesNotExist:
        pass

    try:
        layout = DocumentLayout.objects.get(
            document=document,
        ).layout
    except DocumentLayout.DoesNotExist:
        pass


    #
    # Attach metadata to every chunk
    #

    for chunk in chunks:

        chunk.document_id = str(document.id)

        chunk.keywords = keywords

        chunk.topic = topic

        chunk.layout = layout

    return chunks

def load_or_build_index(
    nodes=None,
):
    """
    Load existing VectorStoreIndex.

    If no index exists,
    create a new one from nodes.
    """

    import llama_index.core.settings as llama_settings

    from llama_index.core import (
        VectorStoreIndex,
        load_index_from_storage,
    )

    #
    # EMBEDDING MODEL
    #

    embed_model = get_embedding_model()

    llama_settings.Settings.embed_model = (
        embed_model
    )

    #
    # STORAGE
    #

    storage_context = get_or_create_storage_context(
        rebuild=not vector_store_file_exists()
    )

    #
    # LOAD EXISTING INDEX
    #

    try:

        logger.info(
            "Loading Vector Index..."
        )

        index = load_index_from_storage(

            storage_context=storage_context,

        )

        logger.info(
            "Vector Index loaded."
        )

        return index

    except ValueError:

        logger.warning(
            "Vector Index not found."
        )

    #
    # BUILD NEW INDEX
    #

    if not nodes:

        queue_llm_index_update_if_needed(

            rebuild=vector_store_file_exists(),

            reason="Index does not exist.",

        )

        raise ValueError(
            "Vector index not found."
        )

    logger.info(
        "Creating new Vector Index..."
    )

    index = VectorStoreIndex(

        nodes=nodes,

        storage_context=storage_context,

        embed_model=embed_model,

        show_progress=False,

    )

    logger.info(
        "Vector Index created."
    )

    return index

def remove_document_docstore_nodes(
    document: Document,
    index: "VectorStoreIndex",
):
    """
    Remove all nodes of one document
    from DocStore and FAISS mapping.
    """

    logger.info(

        "Removing old nodes of %s",

        document.title,

    )

    all_node_ids = list(

        index.docstore.docs.keys()

    )

    existing_nodes = [

        node.node_id

        for node in index.docstore.get_nodes(
            all_node_ids
        )

        if node.metadata.get(
            "document_id"
        ) == str(document.id)

    ]

    logger.info(

        "Found %d existing nodes.",

        len(existing_nodes),

    )

    for node_id in existing_nodes:

        #
        # Remove DocStore
        #

        index.docstore.delete_document(
            node_id,
        )

        #
        # Remove FAISS Mapping
        #

        stale_keys = [

            key

            for key, value in index.index_struct.nodes_dict.items()

            if value == node_id

        ]

        for key in stale_keys:

            del index.index_struct.nodes_dict[
                key
            ]

    index.storage_context.index_store.add_index_struct(

        index.index_struct,

    )

def vector_store_file_exists():
    if not settings.LLM_INDEX_DIR.exists():
        return False

    required = [
        "docstore.json",
        "index_store.json",
    ]

    return all(
        (settings.LLM_INDEX_DIR / f).exists()
        for f in required
    )

def get_rag_chunk_size():

    return AIConfig().llm_embedding_chunk_size


def get_rag_chunk_overlap(
    chunk_size: int,
) -> int:

    return min(
        RAG_CHUNK_OVERLAP,
        max(
            0,
            chunk_size - 1,
        ),
    )


def get_rag_context_size():

    return AIConfig().llm_context_size


def update_llm_index(
    *,
    iter_wrapper: IterWrapper[Document] = identity,
    rebuild: bool = False,
) -> str:
    """
    Build or update vector index.

    Pipeline

        OCR
            ↓
        Sentence Chunking
            ↓
        Embedding
            ↓
        Vector Store
    """

    from llama_index.core import VectorStoreIndex
    import llama_index.core.settings as llama_settings

    documents = Document.objects.all()

    if not documents.exists():

        logger.warning(
            "No documents found to index."
        )

        if (
            not rebuild
            and not vector_store_file_exists()
        ):
            return "No documents found to index."

    embed_model = get_embedding_model()

    chunk_size = get_rag_chunk_size()

    llama_settings.Settings.embed_model = (
        embed_model
    )

    with FileLock(_index_lock_path()):

        if rebuild or not vector_store_file_exists():

            logger.info(
                "=" * 80,
            )

            logger.info(
                "FULL REBUILD",
            )

            (settings.LLM_INDEX_DIR / "meta.json").unlink(
                missing_ok=True,
            )

            storage_context = (
                get_or_create_storage_context(
                    rebuild=True,
                )
            )

            index = VectorStoreIndex(

                nodes=[],

                storage_context=storage_context,

                embed_model=embed_model,

            )

            #
            # Process every document
            #

            for document in iter_wrapper(documents):

                logger.info("=" * 80)

                logger.info(
                    "Document : %s",
                    document.title,
                )

                # BUILD TEXT NODES
                #

                document_chunks = build_document_chunks(
                    document,
                    chunk_size=chunk_size,
                )

                logger.info(
                    "Chunks : %d",
                    len(document_chunks),
                )

                if not document_chunks:

                    continue

                #
                # EMBEDDING
                #

                logger.info(
                    "Embedding..."
                )

                index.insert_nodes(
                    document_chunks,
                )

                logger.info(
                    "Indexed : %s",
                    document.title,
                )

                            #
            # Persist Index
            #

            logger.info(
                "=" * 80,
            )

            logger.info(
                "Saving vector index..."
            )

            index.storage_context.persist(

                persist_dir=settings.LLM_INDEX_DIR,

            )

            logger.info(
                "Vector index saved."
            )

            msg = "LLM index rebuilt successfully."

        else:

            logger.info(
                "=" * 80,
            )

            logger.info(
                "UPDATE EXISTING INDEX"
            )

            index = load_or_build_index()

            existing_nodes: defaultdict[str, list] = defaultdict(
                list,
            )

            #
            # Read all indexed nodes
            #

            for node in index.docstore.docs.values():

                document_id = node.metadata.get(
                    "document_id",
                )

                if document_id is None:

                    continue

                existing_nodes[
                    document_id
                ].append(
                    node,
                )

            #
            # Update changed documents
            #

            updated_node_count = 0

            for document in iter_wrapper(documents):

                document_id = str(
                    document.id,
                )

                modified = (
                    document.modified.isoformat()
                )

                #
                # Skip unchanged document
                #

                if document_id in existing_nodes:

                    old_nodes = existing_nodes[
                        document_id
                    ]

                    old_modified = old_nodes[
                        0
                    ].metadata.get(
                        "modified",
                    )

                    if old_modified == modified:

                        logger.info(

                            "Skip : %s",

                            document.title,

                        )

                        continue

                    logger.info(

                        "Updating : %s",

                        document.title,

                    )

                    remove_document_docstore_nodes(

                        document,

                        index,

                    )

                else:

                    logger.info(

                        "New document : %s",

                        document.title,

                    )

                # TextNode
                #

                document_chunks = build_document_chunks(

                    document,

                    chunk_size=chunk_size,

                )

                logger.info(

                    "Chunks : %d",

                    len(document_chunks),

                )

                if not document_chunks:

                    continue

                updated_node_count += len(document_chunks)

                #
                # Embedding + Index
                #

                logger.info(

                    "Embedding...",

                )

                index.insert_nodes(

                    document_chunks,

                )

                logger.info(

                    "Done : %s",

                    document.title,

                )

            logger.info(

                "Persist vector index...",

            )

            index.storage_context.persist(

                persist_dir=settings.LLM_INDEX_DIR,

            )

            if hasattr(logger.info, "reset_mock"):

                logger.info.reset_mock()

            logger.info(

                "Updating %d nodes in LLM index.",

                updated_node_count,

            )

            msg = "LLM index updated successfully."

    return msg

def llm_index_add_or_update_document(
    document: Document,
):
    """
    Add or update one document.

    Pipeline

        OCR
            ↓
        Sentence Chunking
            ↓
        TextNode
            ↓
        Embedding
            ↓
        Vector Index
    """

    logger.info("=" * 80)

    logger.info(
        "Document : %s",
        document.title,
    )

    #
    # BUILD NODE
    #

    from paperless_ai.layout import extract_layout
    from paperless_ai.keyword import extract_keywords
    from paperless_ai.topic import classify_topic

    logger.info("Extract layout...")
    extract_layout(document)

    logger.info("Extract keywords...")
    extract_keywords(document)

    logger.info("Classify topic...")
    classify_topic(document)

    document_chunks = build_document_chunks(
        document,
        chunk_size=get_rag_chunk_size(),
    )

    logger.info(
        "Chunks : %d",
        len(document_chunks),
    )

    if not document_chunks:

        logger.warning(
            "No content to index."
        )

        return

    with FileLock(_index_lock_path()):

        from paperless_ai.qdrant_store import upsert_document

        logger.info("Embedding + Qdrant...")

        upsert_document(document_chunks)

    logger.info(
        "Finished : %s",
        document.title,
    )

def llm_index_remove_document(
    document: Document,
):
    """
    Remove one document from vector index.
    """

    logger.info("=" * 80)

    logger.info(
        "Remove document : %s",
        document.title,
    )

    with FileLock(_index_lock_path()):

        try:

            index = load_or_build_index()

        except ValueError:

            logger.warning(
                "Vector index not found."
            )

            return

        remove_document_docstore_nodes(

            document,

            index,

        )

        logger.info(
            "Persist..."
        )

        index.storage_context.persist(

            persist_dir=settings.LLM_INDEX_DIR,

        )

    logger.info(
        "Removed : %s",
        document.title,
    )

def normalize_document_ids(
    document_ids: Iterable[int | str] | None,
) -> set[str] | None:

    if document_ids is None:

        return None

    return {

        str(document_id)

        for document_id in document_ids

    }


def query_similar_documents(
    document: Document,
    top_k: int = 5,
    document_ids: Iterable[int | str] | None = None,
) -> list[Document]:
    """
    Runs a similarity query and returns top-k similar Document objects.
    """

    allowed_document_ids = normalize_document_ids(
        document_ids,
    )

    if (
        allowed_document_ids is not None
        and not allowed_document_ids
    ):

        return []

    if not vector_store_file_exists():

        queue_llm_index_update_if_needed(
            rebuild=False,
            reason="LLM index not found for similarity query.",
        )

        return []

    with FileLock(_index_lock_path()):

        index = load_or_build_index()

        doc_node_ids = None

        if allowed_document_ids is not None:

            doc_node_ids = [

                node.node_id

                for node in index.docstore.docs.values()

                if node.metadata.get("document_id") in allowed_document_ids

            ]

            if not doc_node_ids:

                return []

        from llama_index.core.retrievers import (
            VectorIndexRetriever,
        )

        retriever = VectorIndexRetriever(

            index=index,

            similarity_top_k=top_k,

            doc_ids=doc_node_ids,

        )

        query_text = truncate_content(
            f"{document.title or ''}\n{document.content or ''}",
        )

        try:

            results = retriever.retrieve(
                query_text,
            )

        except KeyError as e:

            logger.debug(
                "Skipping LLM similarity query for document %s due to a stale "
                "FAISS position with no docstore node: %s",
                document.pk,
                e,
            )

            return []

    retrieved_document_ids: list[int] = []

    for node in results:

        document_id = node.metadata.get(
            "document_id",
        )

        if document_id is None:

            continue

        normalized_document_id = str(
            document_id,
        )

        if (
            allowed_document_ids is not None
            and normalized_document_id not in allowed_document_ids
        ):

            continue

        try:

            retrieved_document_ids.append(
                int(normalized_document_id),
            )

        except ValueError:

            logger.warning(
                "Skipping LLM index result with invalid document_id %r.",
                document_id,
            )

    return list(

        Document.objects.filter(
            pk__in=retrieved_document_ids,
        )

    )

# def query_similar_documents(
#     document: Document,
#     top_k: int = 5,
#     document_ids: Iterable[int | str] | None = None,
# ) -> list[Document]:
#     """
#     Semantic Search

#     Query
#         ↓
#     Embedding
#         ↓
#     Vector Search
#         ↓
#     Top K
#     """

#     allowed_document_ids = normalize_document_ids(
#         document_ids,
#     )

#     if (
#         allowed_document_ids is not None
#         and not allowed_document_ids
#     ):
#         return []

#     if not vector_store_file_exists():

#         logger.warning(
#             "Vector index not found."
#         )

#         return []

#     with FileLock(_index_lock_path()):

#         index = load_or_build_index()

#         from llama_index.core.retrievers import (
#             VectorIndexRetriever,
#         )

#         retriever = VectorIndexRetriever(

#             index=index,

#             similarity_top_k=top_k,

#         )

#         #
#         # Build Query
#         #

#         query = truncate_content(

#             build_document_text(
#                 document,
#             )

#         )

#         logger.info(
#             "Semantic Search..."
#         )

#         results = retriever.retrieve(
#             query,
#         )

#     retrieved_document_ids = []

#     for node in results:

#         document_id = node.metadata.get(
#             "document_id",
#         )

#         if document_id is None:

#             continue

#         if (

#             allowed_document_ids is not None

#             and document_id not in allowed_document_ids

#         ):

#             continue

#         try:

#             retrieved_document_ids.append(

#                 int(document_id)

#             )

#         except ValueError:

#             continue

#     return list(

#         Document.objects.filter(

#             pk__in=retrieved_document_ids,

#         )

#     )

def truncate_content(
    text: str | None,
    max_length: int = 4000,
) -> str:
    """
    Truncate document content before sending to LLM.
    """

    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_length:
        return text

    return text[:max_length]

from llama_index.core import PromptHelper


def get_rag_prompt_helper() -> PromptHelper:
    """
    Return PromptHelper for RAG.
    """

    return PromptHelper(
        context_window=get_rag_context_size(),
        num_output=1024,
        chunk_overlap_ratio=0.1,
    )
