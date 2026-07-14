from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.db.models import Case
from django.db.models import IntegerField
from django.db.models import Value
from django.db.models import When

from documents.models import Document
from documents.permissions import get_objects_for_user_owner_aware
from documents.search._hybrid_fulltext import score_hybrid_fulltext_document
from documents.search._query_preprocessor import PreprocessedQuery
from documents.search._query_preprocessor import preprocess_hybrid_query
from paperless_ai.semantic_search import SemanticSearchHit
from paperless_ai.semantic_search import semantic_search

if TYPE_CHECKING:
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.db.models import QuerySet


DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_FULLTEXT_WEIGHT = 0.65
DEFAULT_SEMANTIC_WEIGHT = 0.35
DEFAULT_CHANNEL_AGREEMENT_BONUS = 0.05


@dataclass(frozen=True, slots=True)
class HybridSearchHit:
    document_id: int
    score: float
    rank: int
    title: str
    fulltext_score: float
    semantic_score: float
    fulltext_rank: int | None
    semantic_rank: int | None
    matched_channels: tuple[str, ...]
    explanation: dict[str, object]


@dataclass(frozen=True, slots=True)
class HybridSearchResults:
    query: PreprocessedQuery
    hits: list[HybridSearchHit]
    total: int

    @property
    def document_ids(self) -> list[int]:
        return [hit.document_id for hit in self.hits]


@dataclass(frozen=True, slots=True)
class _FullTextCandidate:
    document_id: int
    score: float
    rank: int
    raw_score: float
    best_match_type: str | None
    best_match_field: str | None
    match_count: int


@dataclass(frozen=True, slots=True)
class _HybridCandidate:
    document_id: int
    raw_score: float
    title: str
    fulltext_score: float
    semantic_score: float
    fulltext_rank: int | None
    semantic_rank: int | None
    matched_channels: tuple[str, ...]
    explanation: dict[str, object]


def hybrid_search(
    query: str,
    *,
    user: AbstractBaseUser | None = None,
    top_k: int = DEFAULT_TOP_K,
    fulltext_weight: float = DEFAULT_FULLTEXT_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    channel_agreement_bonus: float = DEFAULT_CHANNEL_AGREEMENT_BONUS,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    min_score: float | None = None,
) -> HybridSearchResults:
    """
    Run full-text + semantic hybrid search and re-rank document candidates.

    Both channels are normalized to 0..1 before fusion:
    - full-text raw mentor scores are divided by the best full-text score
    - semantic scores are already normalized by ``semantic_search``

    The final score is a weighted sum plus a small agreement bonus when both
    channels find the same document, then normalized so the top hit is 1.0.
    """
    prepared_query = preprocess_hybrid_query(query)
    if top_k <= 0 or not prepared_query.normalized_query:
        return HybridSearchResults(query=prepared_query, hits=[], total=0)

    candidate_limit = max(top_k * max(candidate_multiplier, 1), top_k)
    fulltext_candidates = _fulltext_candidates(
        query,
        user=user,
        limit=candidate_limit,
    )
    semantic_candidates = semantic_search(
        query,
        user=user,
        top_k=candidate_limit,
    ).hits

    fused_candidates = _fuse_candidates(
        fulltext_candidates=fulltext_candidates,
        semantic_candidates=semantic_candidates,
        fulltext_weight=fulltext_weight,
        semantic_weight=semantic_weight,
        channel_agreement_bonus=channel_agreement_bonus,
    )
    fused_candidates = _normalize_hybrid_scores(fused_candidates)

    if min_score is not None:
        fused_candidates = [
            candidate
            for candidate in fused_candidates
            if candidate.raw_score >= min_score
        ]

    hits = [
        HybridSearchHit(
            document_id=candidate.document_id,
            score=candidate.raw_score,
            rank=rank,
            title=candidate.title,
            fulltext_score=candidate.fulltext_score,
            semantic_score=candidate.semantic_score,
            fulltext_rank=candidate.fulltext_rank,
            semantic_rank=candidate.semantic_rank,
            matched_channels=candidate.matched_channels,
            explanation=candidate.explanation,
        )
        for rank, candidate in enumerate(fused_candidates[:top_k], start=1)
    ]

    return HybridSearchResults(
        query=prepared_query,
        hits=hits,
        total=len(hits),
    )


def hybrid_search_queryset(
    query: str,
    *,
    user: AbstractBaseUser | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> QuerySet[Document]:
    """
    Convenience wrapper returning a Document queryset ordered by hybrid rank.
    """
    results = hybrid_search(
        query,
        user=user,
        top_k=top_k,
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


def _fulltext_candidates(
    query: str,
    *,
    user: AbstractBaseUser | None,
    limit: int,
) -> dict[int, _FullTextCandidate]:
    preprocessed_query = preprocess_hybrid_query(query)
    if not preprocessed_query.normalized_query:
        return {}

    scored_results = []
    for document in _candidate_documents(user):
        scored_result = score_hybrid_fulltext_document(
            document_id=document.pk,
            fields={
                "document_code": _document_code(document),
                "title": document.title,
                "content": document.content,
            },
            preprocessed_query=preprocessed_query,
        )
        if scored_result is not None:
            scored_results.append(scored_result)

    raw_results = sorted(
        scored_results,
        key=lambda result: (-result.score, result.document_id),
    )[:limit]

    max_score = max(
        (result.score for result in raw_results),
        default=0.0,
    )
    if max_score <= 0.0:
        return {}

    candidates: dict[int, _FullTextCandidate] = {}
    for rank, result in enumerate(raw_results, start=1):
        best_match = result.matches[0] if result.matches else None
        candidates[result.document_id] = _FullTextCandidate(
            document_id=result.document_id,
            score=result.score / max_score,
            rank=rank,
            raw_score=result.score,
            best_match_type=best_match.match_type if best_match else None,
            best_match_field=best_match.field if best_match else None,
            match_count=len(result.matches),
        )
    return candidates


def _candidate_documents(user: AbstractBaseUser | None):
    if user is None or getattr(user, "is_superuser", False):
        return Document.objects.all().only(
            "id",
            "title",
            "content",
            "checksum",
            "archive_serial_number",
        )

    return get_objects_for_user_owner_aware(
        user,
        "view_document",
        Document,
    ).only(
        "id",
        "title",
        "content",
        "checksum",
        "archive_serial_number",
    )


def _document_code(document: Document) -> str:
    parts = [str(document.pk), document.checksum]
    if document.archive_serial_number is not None:
        parts.append(str(document.archive_serial_number))
    return " ".join(parts)


def _fuse_candidates(
    *,
    fulltext_candidates: dict[int, _FullTextCandidate],
    semantic_candidates: list[SemanticSearchHit],
    fulltext_weight: float,
    semantic_weight: float,
    channel_agreement_bonus: float,
) -> list[_HybridCandidate]:
    semantic_by_document = {
        hit.document_id: hit for hit in semantic_candidates
    }
    document_ids = set(fulltext_candidates) | set(semantic_by_document)
    titles = _document_titles(document_ids)

    candidates: list[_HybridCandidate] = []
    for document_id in document_ids:
        fulltext = fulltext_candidates.get(document_id)
        semantic = semantic_by_document.get(document_id)
        fulltext_score = fulltext.score if fulltext else 0.0
        semantic_score = semantic.score if semantic else 0.0
        matched_channels = tuple(
            channel
            for channel, score in (
                ("fulltext", fulltext_score),
                ("semantic", semantic_score),
            )
            if score > 0.0
        )

        agreement_bonus = (
            channel_agreement_bonus
            if fulltext_score > 0.0 and semantic_score > 0.0
            else 0.0
        )
        score = (
            fulltext_weight * fulltext_score
            + semantic_weight * semantic_score
            + agreement_bonus
        )

        candidates.append(
            _HybridCandidate(
                document_id=document_id,
                raw_score=score,
                title=titles.get(document_id, semantic.title if semantic else ""),
                fulltext_score=fulltext_score,
                semantic_score=semantic_score,
                fulltext_rank=fulltext.rank if fulltext else None,
                semantic_rank=semantic.rank if semantic else None,
                matched_channels=matched_channels,
                explanation={
                    "fulltext_weight": fulltext_weight,
                    "semantic_weight": semantic_weight,
                    "channel_agreement_bonus": agreement_bonus,
                    "fulltext_raw_score": fulltext.raw_score if fulltext else 0.0,
                    "fulltext_best_match_type": (
                        fulltext.best_match_type if fulltext else None
                    ),
                    "fulltext_best_match_field": (
                        fulltext.best_match_field if fulltext else None
                    ),
                    "fulltext_match_count": fulltext.match_count if fulltext else 0,
                    "semantic_matched_nodes": (
                        semantic.matched_nodes if semantic else 0
                    ),
                },
            ),
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.raw_score,
            candidate.fulltext_score,
            candidate.semantic_score,
        ),
        reverse=True,
    )


def _normalize_hybrid_scores(
    candidates: list[_HybridCandidate],
) -> list[_HybridCandidate]:
    max_score = max((candidate.raw_score for candidate in candidates), default=0.0)
    if max_score <= 0.0:
        return candidates

    normalized = [
        _HybridCandidate(
            document_id=candidate.document_id,
            raw_score=candidate.raw_score / max_score,
            title=candidate.title,
            fulltext_score=candidate.fulltext_score,
            semantic_score=candidate.semantic_score,
            fulltext_rank=candidate.fulltext_rank,
            semantic_rank=candidate.semantic_rank,
            matched_channels=candidate.matched_channels,
            explanation=candidate.explanation,
        )
        for candidate in candidates
    ]
    return sorted(
        normalized,
        key=lambda candidate: (
            candidate.raw_score,
            candidate.fulltext_score,
            candidate.semantic_score,
        ),
        reverse=True,
    )


def _document_titles(document_ids: set[int]) -> dict[int, str]:
    if not document_ids:
        return {}
    return dict(
        Document.objects.filter(pk__in=document_ids).values_list("pk", "title"),
    )
