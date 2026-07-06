from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class PreprocessedQuery:
    original_query: str
    normalized_query: str
    tokens: list[str]
    phrase_groups: list[str]


def normalize_hybrid_query(query: str) -> str:
    """
    Normalize free-text input before hybrid/full-text search.

    Keeps Vietnamese accents intact while making spacing and case consistent.
    """
    normalized = unicodedata.normalize("NFC", query)
    normalized = normalized.strip().lower()
    return _WHITESPACE_RE.sub(" ", normalized)


def tokenize_query(normalized_query: str) -> list[str]:
    """Split a normalized query into word tokens."""
    if not normalized_query:
        return []
    return [token for token in normalized_query.split(" ") if token]


def build_phrase_groups(tokens: list[str]) -> list[str]:
    """
    Build prefix phrase groups for scoring full-text matches.

    Example:
        ["chien", "luoc", "ai"] -> ["chien", "chien luoc", "chien luoc ai"]
    """
    return [" ".join(tokens[:index]) for index in range(1, len(tokens) + 1)]


def preprocess_hybrid_query(query: str) -> PreprocessedQuery:
    """
    Prepare user input for hybrid search.

    The result is intended for:
    - exact search with ``normalized_query``
    - token search with ``tokens``
    - phrase search with ``phrase_groups``
    """
    normalized_query = normalize_hybrid_query(query)
    tokens = tokenize_query(normalized_query)
    phrase_groups = build_phrase_groups(tokens)

    return PreprocessedQuery(
        original_query=query,
        normalized_query=normalized_query,
        tokens=tokens,
        phrase_groups=phrase_groups,
    )
