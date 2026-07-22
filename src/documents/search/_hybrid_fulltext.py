from __future__ import annotations

from dataclasses import dataclass

from documents.search._query_preprocessor import PreprocessedQuery
from documents.search._tokenizer import ascii_fold


@dataclass(frozen=True, slots=True)
class HybridFullTextMatch:
    field: str
    match_type: str
    matched_text: str
    score: float


@dataclass(frozen=True, slots=True)
class HybridFullTextResult:
    document_id: int
    score: float
    matches: list[HybridFullTextMatch]


def _normalize_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item is not None)
    return str(value).strip().lower()


def _contains(value: str, query: str) -> bool:
    if not value or not query:
        return False
    if query in value:
        return True
    return ascii_fold(query) in ascii_fold(value)


def _equals(value: str, query: str) -> bool:
    if not value or not query:
        return False
    if value == query:
        return True
    return ascii_fold(value) == ascii_fold(query)


def score_hybrid_fulltext_document(
    document_id: int,
    fields: dict[str, object],
    preprocessed_query: PreprocessedQuery,
) -> HybridFullTextResult | None:
    """
    Score one indexed document using the hybrid full-text strategy.

    Priority follows the mentor's scoring model:
    document code/title exact matches rank highest, then longer phrases,
    then shorter phrases, then single token matches.
    """
    normalized_query = preprocessed_query.normalized_query
    if not normalized_query:
        return None

    searchable_fields = {
        "document_code": _normalize_value(fields.get("document_code")),
        "title": _normalize_value(fields.get("title")),
        "content": _normalize_value(fields.get("content")),
    }

    matches: list[HybridFullTextMatch] = []

    exact_scores = {
        "document_code": 120.0,
        "title": 110.0,
        "content": 90.0,
    }
    for field, value in searchable_fields.items():
        if _equals(value, normalized_query) or _contains(value, normalized_query):
            matches.append(
                HybridFullTextMatch(
                    field=field,
                    match_type="exact_phrase",
                    matched_text=normalized_query,
                    score=exact_scores[field],
                ),
            )

    phrase_groups = [
        phrase
        for phrase in preprocessed_query.phrase_groups
        if phrase
        and phrase != normalized_query
        and len(phrase.split()) > 1
    ]
    for phrase_index, phrase in enumerate(reversed(phrase_groups), start=1):
        phrase_length_bonus = len(phrase.split()) * 5
        for field, value in searchable_fields.items():
            if not _contains(value, phrase):
                continue
            field_base = {
                "document_code": 80.0,
                "title": 70.0,
                "content": 55.0,
            }[field]
            matches.append(
                HybridFullTextMatch(
                    field=field,
                    match_type="phrase_group",
                    matched_text=phrase,
                    score=field_base + phrase_length_bonus - phrase_index,
                ),
            )

    matched_tokens_by_field: dict[str, set[str]] = {
        field: set() for field in searchable_fields
    }
    for token in preprocessed_query.tokens:
        for field, value in searchable_fields.items():
            if not _contains(value, token):
                continue
            matched_tokens_by_field[field].add(token)
            token_score = {
                "document_code": 12.0,
                "title": 8.0,
                "content": 4.0,
            }[field]
            matches.append(
                HybridFullTextMatch(
                    field=field,
                    match_type="token",
                    matched_text=token,
                    score=token_score,
                ),
            )

    if len(preprocessed_query.tokens) > 1:
        has_phrase_match = any(
            match.match_type in {"exact_phrase", "phrase_group"}
            for match in matches
        )
        has_complete_token_match = any(
            len(tokens) == len(set(preprocessed_query.tokens))
            for tokens in matched_tokens_by_field.values()
        )
        if not has_phrase_match and not has_complete_token_match:
            return None

    if not matches:
        return None

    return HybridFullTextResult(
        document_id=document_id,
        score=sum(match.score for match in matches),
        matches=sorted(matches, key=lambda match: match.score, reverse=True),
    )
