from __future__ import annotations

import pytest

from documents.models import Document
from documents.search._query_preprocessor import preprocess_hybrid_query
from paperless_ai import hybrid_search as hybrid_search_module
from paperless_ai.hybrid_search import hybrid_search
from paperless_ai.semantic_search import SemanticSearchHit
from paperless_ai.semantic_search import SemanticSearchResults

pytestmark = pytest.mark.django_db


def test_hybrid_search_returns_all_fulltext_matches_by_default(monkeypatch) -> None:
    Document.objects.create(
        title="Chiến lược phát triển AI",
        content="Chiến lược phát triển AI trong vận hành.",
        checksum="A",
        pk=1,
    )
    Document.objects.create(
        title="Kế hoạch AI",
        content="Ứng dụng AI trong nghiệp vụ.",
        checksum="B",
        pk=2,
    )

    def _semantic_search(*args, **kwargs):
        return SemanticSearchResults(
            query=preprocess_hybrid_query(args[0]),
            hits=[],
            total=0,
        )

    monkeypatch.setattr(hybrid_search_module, "semantic_search", _semantic_search)

    results = hybrid_search("chiến lược phát triển AI")

    assert results.total == 2
    assert results.document_ids == [1, 2]


def test_hybrid_search_limits_only_when_requested(monkeypatch) -> None:
    Document.objects.create(
        title="Chiến lược phát triển AI",
        content="Chiến lược phát triển AI trong vận hành.",
        checksum="A",
        pk=1,
    )
    Document.objects.create(
        title="Kế hoạch AI",
        content="Ứng dụng AI trong nghiệp vụ.",
        checksum="B",
        pk=2,
    )

    def _semantic_search(*args, **kwargs):
        return SemanticSearchResults(
            query=preprocess_hybrid_query(args[0]),
            hits=[],
            total=0,
        )

    monkeypatch.setattr(hybrid_search_module, "semantic_search", _semantic_search)

    results = hybrid_search("chiến lược phát triển AI", top_k=1)

    assert results.total == 1
    assert results.document_ids == [1]


def test_hybrid_search_restricts_semantic_to_fulltext_candidates(monkeypatch) -> None:
    Document.objects.create(
        title="Chiến lược phát triển AI",
        content="Chiến lược phát triển AI trong vận hành.",
        checksum="A",
        pk=1,
    )
    Document.objects.create(
        title="Semantic only unrelated",
        content="Nội dung không khớp truy vấn.",
        checksum="B",
        pk=2,
    )

    captured_document_ids = None

    def _semantic_search(*args, **kwargs):
        nonlocal captured_document_ids
        captured_document_ids = kwargs["document_ids"]
        return SemanticSearchResults(
            query=preprocess_hybrid_query(args[0]),
            hits=[
                SemanticSearchHit(
                    document_id=2,
                    score=1.0,
                    rank=1,
                    matched_nodes=1,
                    title="Semantic only unrelated",
                    metadata={},
                    samples=[],
                ),
            ],
            total=1,
        )

    monkeypatch.setattr(hybrid_search_module, "semantic_search", _semantic_search)

    results = hybrid_search("chiến lược phát triển AI")

    assert captured_document_ids == [1]
    assert results.document_ids == [1]


def test_hybrid_search_returns_content_summary_samples_and_match_count(
    monkeypatch,
) -> None:
    Document.objects.create(
        title="Chiến lược phát triển AI",
        content=(
            "Mở đầu tài liệu. Chiến lược phát triển AI cần gắn với dữ liệu. "
            "Phần sau tiếp tục nhắc lại chiến lược phát triển AI trong vận hành."
        ),
        checksum="A",
        pk=1,
    )

    def _semantic_search(*args, **kwargs):
        return SemanticSearchResults(
            query=preprocess_hybrid_query(args[0]),
            hits=[],
            total=0,
        )

    monkeypatch.setattr(hybrid_search_module, "semantic_search", _semantic_search)

    result = hybrid_search("chiến lược phát triển AI").hits[0]

    assert "Chiến lược phát triển AI" in result.summary
    assert result.content_samples
    assert "Chiến lược phát triển AI" in result.content_samples[0]
    assert result.content_match_count >= 2
