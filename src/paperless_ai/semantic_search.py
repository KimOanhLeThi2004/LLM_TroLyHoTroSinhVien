from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

from django.db.models import Case
from django.db.models import IntegerField
from django.db.models import Value
from django.db.models import When

from documents.models import Document
from documents.permissions import get_objects_for_user_owner_aware
from documents.search._query_preprocessor import PreprocessedQuery
from documents.search._query_preprocessor import preprocess_hybrid_query
from paperless_ai.indexing import load_or_build_index
from paperless_ai.indexing import queue_llm_index_update_if_needed
from paperless_ai.indexing import vector_store_file_exists

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.db.models import QuerySet
    from llama_index.core.schema import NodeWithScore


logger = logging.getLogger("paperless_ai.semantic_search")

DEFAULT_TOP_K = 10
DEFAULT_NODE_OVERFETCH = 4


@dataclass(frozen=True, slots=True)
class SemanticSearchHit:
    document_id: int
    score: float
    rank: int
    matched_nodes: int
    title: str
    metadata: dict[str, Any]
    samples: list[str]


@dataclass(frozen=True, slots=True)
class SemanticSearchResults:
    query: PreprocessedQuery
    hits: list[SemanticSearchHit]
    total: int

    @property
    def document_ids(self) -> list[int]:
        return [hit.document_id for hit in self.hits]


def semantic_search(
    query: str,
    *,
    user: AbstractBaseUser | None = None,
    top_k: int = DEFAULT_TOP_K,
    document_ids: list[int] | None = None,
    min_score: float | None = None,
    node_overfetch: int = DEFAULT_NODE_OVERFETCH,
) -> SemanticSearchResults:
    """
    Run semantic document search against the Paperless AI FAISS/LLM index.

    The query is normalized with the existing hybrid-search preprocessor, then
    embedded by LlamaIndex and matched against indexed document chunks. Multiple
    matching chunks from the same document are merged into one document-level hit.

    Args:
        query: Raw user query.
        user: Optional user used to enforce owner/object-aware view permissions.
        top_k: Maximum number of documents to return.
        document_ids: Optional allow-list of document IDs to search within.
        min_score: Optional minimum normalized relevance score, from 0 to 1.
        node_overfetch: Number of vector nodes to fetch per requested document.

    Returns:
        SemanticSearchResults with ranked document-level hits. If the index is
        missing, an index update is queued and an empty result is returned.
    """
    prepared_query = preprocess_hybrid_query(query)
    if top_k <= 0 or not prepared_query.normalized_query:
        return SemanticSearchResults(query=prepared_query, hits=[], total=0)

    if not vector_store_file_exists():
        queue_llm_index_update_if_needed(
            rebuild=False,
            reason="LLM index not found for semantic search.",
        )
        return SemanticSearchResults(query=prepared_query, hits=[], total=0)

    index = load_or_build_index()
    candidate_node_ids = _candidate_node_ids(index, document_ids)
    if document_ids and not candidate_node_ids:
        return SemanticSearchResults(query=prepared_query, hits=[], total=0)

    from llama_index.core.retrievers import VectorIndexRetriever

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=max(top_k * max(node_overfetch, 1), top_k),
        doc_ids=candidate_node_ids,
    )

    node_results = retriever.retrieve(_build_semantic_query_text(prepared_query))
    aggregated_hits = _normalize_document_scores(
        _aggregate_node_results(node_results),
    )
    if min_score is not None:
        aggregated_hits = [
            hit for hit in aggregated_hits if hit.score >= min_score
        ]

    permitted_ids = _permitted_document_ids(
        [hit.document_id for hit in aggregated_hits],
        user=user,
    )

    hits = [
        hit
        for hit in aggregated_hits
        if hit.document_id in permitted_ids
    ][:top_k]

    return SemanticSearchResults(
        query=prepared_query,
        hits=[
            SemanticSearchHit(
                document_id=hit.document_id,
                score=hit.score,
                rank=rank,
                matched_nodes=hit.matched_nodes,
                title=hit.title,
                metadata=hit.metadata,
                samples=hit.samples,
            )
            for rank, hit in enumerate(hits, start=1)
        ],
        total=len(hits),
    )


def semantic_search_queryset(
    query: str,
    *,
    user: AbstractBaseUser | None = None,
    top_k: int = DEFAULT_TOP_K,
    document_ids: list[int] | None = None,
    min_score: float | None = None,
) -> QuerySet[Document]:
    """
    Convenience wrapper returning a Document queryset ordered by semantic rank.
    """
    results = semantic_search(
        query,
        user=user,
        top_k=top_k,
        document_ids=document_ids,
        min_score=min_score,
    )
    if not results.document_ids:
        return Document.objects.none()

    preserved_order = Case(
        *[
            When(pk=document_id, then=Value(position))
            for position, document_id in enumerate(results.document_ids)
        ],
        output_field=IntegerField(),
    )
    return Document.objects.filter(pk__in=results.document_ids).order_by(
        preserved_order,
    )


@dataclass(frozen=True, slots=True)
class _AggregatedHit:
    document_id: int
    score: float
    matched_nodes: int
    title: str
    metadata: dict[str, Any]
    samples: list[str]


def _build_semantic_query_text(query: PreprocessedQuery) -> str:
    if query.phrase_groups:
        return "\n".join(
            [
                f"Query: {query.normalized_query}",
                f"Important phrase: {query.phrase_groups[-1]}",
                f"Keywords: {', '.join(query.tokens)}",
            ],
        )
    return query.normalized_query


def _candidate_node_ids(index, document_ids: list[int] | None) -> list[str] | None:
    if not document_ids:
        return None

    allowed_ids = {int(document_id) for document_id in document_ids}
    return [
        node.node_id
        for node in index.docstore.docs.values()
        if _safe_int(node.metadata.get("document_id")) in allowed_ids
    ]


def _aggregate_node_results(
    node_results: list[NodeWithScore],
) -> list[_AggregatedHit]:
    by_document: dict[int, _AggregatedHit] = {}

    for node_result in node_results:
        metadata = dict(node_result.metadata or {})
        document_id = _safe_int(metadata.get("document_id"))
        if document_id is None:
            continue

        score = _normalize_l2_score(node_result.score)

        existing = by_document.get(document_id)
        title = str(metadata.get("title") or "")
        sample = _node_sample(node_result)
        if existing is None:
            by_document[document_id] = _AggregatedHit(
                document_id=document_id,
                score=score,
                matched_nodes=1,
                title=title,
                metadata=metadata,
                samples=[sample] if sample else [],
            )
            continue

        samples = existing.samples
        if sample and sample not in samples:
            samples = [*samples, sample]

        by_document[document_id] = _AggregatedHit(
            document_id=document_id,
            score=max(existing.score, score)
            + _multi_node_boost(existing.matched_nodes),
            matched_nodes=existing.matched_nodes + 1,
            title=existing.title or title,
            metadata=existing.metadata,
            samples=samples[:3],
        )

    return sorted(
        by_document.values(),
        key=lambda hit: (hit.score, hit.matched_nodes),
        reverse=True,
    )


def _normalize_document_scores(hits: list[_AggregatedHit]) -> list[_AggregatedHit]:
    """
    Normalize final document scores so they match Tantivy search score semantics.

    Full-text search returns scores in the 0..1 range with the top hit at 1.0.
    Semantic scores are first computed from FAISS distance and multi-node boosts,
    then normalized the same way so hybrid merging can compare both channels.
    """
    max_score = max((hit.score for hit in hits), default=0.0)
    if max_score <= 0.0:
        return hits

    normalized_hits = [
        _AggregatedHit(
            document_id=hit.document_id,
            score=hit.score / max_score,
            matched_nodes=hit.matched_nodes,
            title=hit.title,
            metadata=hit.metadata,
            samples=hit.samples,
        )
        for hit in hits
    ]
    return sorted(
        normalized_hits,
        key=lambda hit: (hit.score, hit.matched_nodes),
        reverse=True,
    )


def _permitted_document_ids(
    document_ids: list[int],
    *,
    user: AbstractBaseUser | None,
) -> set[int]:
    if not document_ids:
        return set()

    if user is None or not getattr(user, "is_authenticated", False):
        return set(
            Document.objects.filter(
                owner__isnull=True,
                pk__in=document_ids,
            ).values_list("pk", flat=True),
        )

    permitted_documents = get_objects_for_user_owner_aware(
        user,
        "view_document",
        Document,
    ).filter(pk__in=document_ids)
    return set(permitted_documents.values_list("pk", flat=True))


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("Skipping semantic result with invalid document_id=%r", value)
        return None


def _node_sample(node_result: NodeWithScore) -> str:
    node = node_result.node
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        try:
            return " ".join(str(get_content()).split())[:500]
        except TypeError:
            pass
    return " ".join(str(getattr(node, "text", "") or "").split())[:500]


def _multi_node_boost(previous_match_count: int) -> float:
    return min(previous_match_count * 0.01, 0.05)


def _normalize_l2_score(score: float | None) -> float:
    """
    Convert FAISS IndexFlatL2 distance scores into a higher-is-better relevance.
    """
    distance = max(float(score or 0.0), 0.0)
    return 1.0 / (1.0 + distance)
