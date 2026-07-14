from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """
    A semantic chunk generated from a document.
    """

    chunk_id: str
    document_id: str | None
    chunk_index: int

    sentence_start: int
    sentence_end: int

    token_count: int

    text: str


class SentenceChunkGenerator:
    """
    Sentence-based Chunk Generator

    Pipeline

    RAW Text
        ↓
    Normalize
        ↓
    Sentence Split
        ↓
    Merge Sentences
        ↓
    Chunk + Metadata
    """


    SENTENCE_REGEX = r"(?<=[.!?\u3002\uff01\uff1f])\s+"

    # ---------------------------------------------------------

    def __init__(
        self,
        *,
        target_tokens: int | None = None,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        overlap_sentences: int | None = None,
    ) -> None:
        if target_tokens is not None:
            self.TARGET_TOKENS = max(1, target_tokens)
            self.MAX_TOKENS = max(
                self.TARGET_TOKENS,
                max_tokens or int(self.TARGET_TOKENS * 1.2),
            )
            self.MIN_TOKENS = max(
                1,
                min_tokens or min(self.TARGET_TOKENS, self.TARGET_TOKENS // 2),
            )

        if overlap_sentences is not None:
            self.OVERLAP_SENTENCES = max(0, overlap_sentences)

    def generate(
        self,
        text: str,
        document_id: str | None = None,
        book_title: str | None = None,
    ) -> List[Chunk]:

        text = self._normalize(text)

        if not text:
            return []

        sentences = self._split_sentences(text)

        prefix = []

        if document_id:
            prefix.append(f"BOOK_ID: {document_id}")

        if book_title:
            prefix.append(f"BOOK_TITLE: {book_title}")

        if prefix:
            sentences = ["\n".join(prefix)] + sentences

        if not sentences:
            return []

        return self._merge_sentences(
            sentences=sentences,
            document_id=document_id,
        )

    # ---------------------------------------------------------

    def _normalize(
        self,
        text: str,
    ) -> str:

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = re.sub(
            r"\n+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ---------------------------------------------------------

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:

        return [
            s.strip()
            for s in re.split(self.SENTENCE_REGEX, text)
            if s.strip()
        ]
    # ---------------------------------------------------------

    def _estimate_tokens(
        self,
        text: str,
    ) -> int:
        """
        Approximate token count.

        (Can later replace by tokenizer.encode())
        """

        return len(text.split())

    # ---------------------------------------------------------


    # ---------------------------------------------------------



    # ---------------------------------------------------------

    def _merge_sentences(
        self,
        sentences: List[str],
        document_id: str | None,
    ) -> List[Chunk]:

        chunks: List[Chunk] = []

        for i, sentence in enumerate(sentences):

            sentence = sentence.strip()

            if not sentence:
                continue

            chunks.append(

                Chunk(

                    chunk_id=str(uuid.uuid4()),

                    document_id=document_id,

                    chunk_index=i,

                    sentence_start=i,

                    sentence_end=i,

                    token_count=self._estimate_tokens(sentence),

                    text=sentence,

                )

            )

        return chunks
