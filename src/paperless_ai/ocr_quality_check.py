from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict
from dataclasses import dataclass

PASS_THRESHOLD = 0.65
MIN_USEFUL_TEXT_LENGTH = 300
GOOD_TEXT_LENGTH = 2_000
MIN_USEFUL_CHARS_PER_PAGE = 300
GOOD_CHARS_PER_PAGE = 1_000
MAX_EXPECTED_NOISE_RATIO = 0.15
GOOD_VALID_WORD_RATIO = 0.65

TOKEN_REGEX = re.compile(r"[\w\d]+", flags=re.UNICODE)
SENTENCE_REGEX = re.compile(r"[.!?。！？]+")
REPEATED_NOISE_REGEX = re.compile(r"([^\w\d\s])\1{2,}", flags=re.UNICODE)

VIETNAMESE_SIGNALS = {
    "và",
    "của",
    "cho",
    "trong",
    "được",
    "không",
    "người",
    "học",
    "tài",
    "liệu",
}

ENGLISH_SIGNALS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "document",
    "data",
    "information",
}

VIETNAMESE_DIACRITICS = set(
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)


@dataclass(frozen=True)
class OcrQualityMetrics:
    text_length: int
    text_length_score: float
    chars_per_page: float | None
    chars_per_page_score: float
    noise_ratio: float
    noise_score: float
    valid_word_ratio: float
    valid_word_score: float
    detected_language: str
    language_confidence: float
    language_score: float
    structure_score: float


@dataclass(frozen=True)
class OcrQualityResult:
    passed: bool
    score: float
    threshold: float
    grade: str
    recommended_action: str
    reasons: list[str]
    metrics: OcrQualityMetrics

    def as_dict(self) -> dict:
        return asdict(self)


def check_ocr_quality(
    *,
    text: str | None,
    page_count: int | None = None,
    expected_language: str | None = None,
) -> OcrQualityResult:
    normalized_text = normalize_text(text)
    reasons: list[str] = []

    text_length = len(normalized_text)
    text_length_score = score_text_length(text_length)

    chars_per_page = calculate_chars_per_page(text_length, page_count)
    chars_per_page_score = score_chars_per_page(chars_per_page)

    noise_ratio = calculate_noise_ratio(normalized_text)
    noise_score = score_noise(noise_ratio)

    valid_word_ratio = calculate_valid_word_ratio(normalized_text)
    valid_word_score = score_valid_word_ratio(valid_word_ratio)

    detected_language, language_confidence = detect_language(normalized_text)
    language_score = score_language(
        detected_language,
        language_confidence,
        expected_language,
    )

    structure_score = score_structure(normalized_text)

    total_score = (
        (0.30 * text_length_score)
        + (0.20 * chars_per_page_score)
        + (0.20 * noise_score)
        + (0.15 * valid_word_score)
        + (0.10 * language_score)
        + (0.05 * structure_score)
    )
    total_score = round4(clamp01(total_score))
    passed = total_score >= PASS_THRESHOLD

    add_reasons(
        reasons=reasons,
        text_length=text_length,
        chars_per_page=chars_per_page,
        noise_ratio=noise_ratio,
        valid_word_ratio=valid_word_ratio,
        detected_language=detected_language,
        language_confidence=language_confidence,
        structure_score=structure_score,
        passed=passed,
    )

    return OcrQualityResult(
        passed=passed,
        score=total_score,
        threshold=PASS_THRESHOLD,
        grade=to_grade(total_score),
        recommended_action="USE_PAPERLESS_CONTENT" if passed else "GEMINI_FALLBACK",
        reasons=reasons,
        metrics=OcrQualityMetrics(
            text_length=text_length,
            text_length_score=round4(text_length_score),
            chars_per_page=round4(chars_per_page) if chars_per_page is not None else None,
            chars_per_page_score=round4(chars_per_page_score),
            noise_ratio=round4(noise_ratio),
            noise_score=round4(noise_score),
            valid_word_ratio=round4(valid_word_ratio),
            valid_word_score=round4(valid_word_score),
            detected_language=detected_language,
            language_confidence=round4(language_confidence),
            language_score=round4(language_score),
            structure_score=round4(structure_score),
        ),
    )


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def score_text_length(length: int) -> float:
    if length <= 0:
        return 0
    if length < MIN_USEFUL_TEXT_LENGTH:
        return 0.5 * length / MIN_USEFUL_TEXT_LENGTH
    return 0.5 + 0.5 * clamp01(
        (length - MIN_USEFUL_TEXT_LENGTH)
        / (GOOD_TEXT_LENGTH - MIN_USEFUL_TEXT_LENGTH),
    )


def calculate_chars_per_page(text_length: int, page_count: int | None) -> float | None:
    if page_count is None or page_count <= 0:
        return None
    return text_length / page_count


def score_chars_per_page(chars_per_page: float | None) -> float:
    if chars_per_page is None:
        return 0.75
    if chars_per_page <= 0:
        return 0
    if chars_per_page < MIN_USEFUL_CHARS_PER_PAGE:
        return 0.5 * chars_per_page / MIN_USEFUL_CHARS_PER_PAGE
    return 0.5 + 0.5 * clamp01(
        (chars_per_page - MIN_USEFUL_CHARS_PER_PAGE)
        / (GOOD_CHARS_PER_PAGE - MIN_USEFUL_CHARS_PER_PAGE),
    )


def calculate_noise_ratio(text: str) -> float:
    if not text:
        return 1

    noisy_characters = 0
    for char in text:
        if char == "\uFFFD" or unicodedata.category(char).startswith("C"):
            noisy_characters += 1
            continue

        category = unicodedata.category(char)
        if category in {"Sm", "Sc", "Sk", "So", "Co", "Cs", "Cn"}:
            noisy_characters += 1

    repeated_noise_penalty = sum(
        max(0, len(match.group(0)) - 2)
        for match in REPEATED_NOISE_REGEX.finditer(text)
    )
    return clamp01((noisy_characters + repeated_noise_penalty) / len(text))


def score_noise(noise_ratio: float) -> float:
    return clamp01(1 - (noise_ratio / MAX_EXPECTED_NOISE_RATIO))


def calculate_valid_word_ratio(text: str) -> float:
    tokens = TOKEN_REGEX.findall(text)
    if not tokens:
        return 0

    valid_words = sum(1 for token in tokens if is_valid_word(token))
    return clamp01(valid_words / len(tokens))


def is_valid_word(token: str) -> bool:
    if len(token) < 2:
        return False

    letter_count = sum(1 for char in token if char.isalpha())
    if letter_count == 0:
        return False
    if letter_count / len(token) < 0.5:
        return False

    lowered = token.lower()
    most_repeated_ratio = max(lowered.count(char) / len(token) for char in set(lowered))
    return most_repeated_ratio < 0.75


def score_valid_word_ratio(valid_word_ratio: float) -> float:
    return clamp01(valid_word_ratio / GOOD_VALID_WORD_RATIO)


def detect_language(text: str) -> tuple[str, float]:
    if not text:
        return "unknown", 0

    lower_text = text.lower()
    tokens = TOKEN_REGEX.findall(lower_text)[:2_000]
    if not tokens:
        return "unknown", 0

    vietnamese_diacritics = sum(1 for char in lower_text if char in VIETNAMESE_DIACRITICS)
    vietnamese_signal_count = sum(1 for token in tokens if token in VIETNAMESE_SIGNALS)
    english_signal_count = sum(1 for token in tokens if token in ENGLISH_SIGNALS)

    vietnamese_score = (vietnamese_diacritics / max(1, len(text))) * 20
    vietnamese_score += (vietnamese_signal_count / len(tokens)) * 4
    english_score = (english_signal_count / len(tokens)) * 5

    if vietnamese_score < 0.03 and english_score < 0.03:
        return "unknown", 0.2
    if vietnamese_score >= english_score:
        return "vi", clamp01(0.45 + vietnamese_score)
    return "en", clamp01(0.45 + english_score)


def score_language(
    detected_language: str,
    confidence: float,
    expected_language: str | None,
) -> float:
    if detected_language == "unknown":
        return confidence
    if not expected_language:
        return confidence

    expected = expected_language.strip().lower()
    if expected.startswith(detected_language):
        return confidence
    return min(0.35, confidence)


def score_structure(text: str) -> float:
    if not text:
        return 0

    raw_lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 2]
    normalized_lines = raw_lines or [
        line.strip() for line in text.split(".") if len(line.strip()) > 12
    ]

    sentence_count = len(SENTENCE_REGEX.findall(text))
    tokens = TOKEN_REGEX.findall(text)
    average_token_length = (
        sum(len(token) for token in tokens) / len(tokens)
        if tokens
        else 0
    )

    line_score = clamp01(len(normalized_lines) / 8)
    sentence_score = clamp01(sentence_count / 8)
    token_length_score = 1 if 3 <= average_token_length <= 12 else 0.35

    return clamp01(
        (0.40 * line_score)
        + (0.40 * sentence_score)
        + (0.20 * token_length_score),
    )


def add_reasons(
    *,
    reasons: list[str],
    text_length: int,
    chars_per_page: float | None,
    noise_ratio: float,
    valid_word_ratio: float,
    detected_language: str,
    language_confidence: float,
    structure_score: float,
    passed: bool,
) -> None:
    if text_length == 0:
        reasons.append("Content is empty.")
    elif text_length < MIN_USEFUL_TEXT_LENGTH:
        reasons.append(
            f"Text length is short ({text_length} chars, minimum target {MIN_USEFUL_TEXT_LENGTH}).",
        )
    else:
        reasons.append(f"Text length is acceptable ({text_length} chars).")

    if chars_per_page is not None and chars_per_page < MIN_USEFUL_CHARS_PER_PAGE:
        reasons.append(f"Characters per page is low ({round4(chars_per_page)} chars/page).")

    if noise_ratio > MAX_EXPECTED_NOISE_RATIO:
        reasons.append(f"OCR noise ratio is high ({round4(noise_ratio)}).")

    if valid_word_ratio < 0.45:
        reasons.append(f"Valid word ratio is low ({round4(valid_word_ratio)}).")

    if detected_language == "unknown" or language_confidence < 0.45:
        reasons.append("Language signal is weak or unknown.")

    if structure_score < 0.35:
        reasons.append(f"Structure signal is weak ({round4(structure_score)}).")

    reasons.append(
        "Paperless OCR content is good enough for indexing."
        if passed
        else "Paperless OCR content should fall back to Gemini OCR/extraction.",
    )


def to_grade(score: float) -> str:
    if score >= 0.80:
        return "GOOD"
    if score >= PASS_THRESHOLD:
        return "MEDIUM"
    return "BAD"


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0
    return max(0, min(1, value))


def round4(value: float) -> float:
    return round(value, 4)
