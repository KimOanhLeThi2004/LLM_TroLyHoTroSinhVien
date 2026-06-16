from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("paperless_ai.gemini_document_extractor")

GEMINI_API_KEY_ENV = "PAPERLESS_GEMINI_API_KEY"
GEMINI_MODEL_ENV = "PAPERLESS_GEMINI_MODEL"
GEMINI_ENDPOINT_ENV = "PAPERLESS_GEMINI_ENDPOINT"
GEMINI_MAX_BYTES_ENV = "PAPERLESS_GEMINI_MAX_INLINE_BYTES"
GEMINI_TIMEOUT_ENV = "PAPERLESS_GEMINI_TIMEOUT_SECONDS"

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MAX_INLINE_BYTES = 20 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 300.0


class GeminiExtractorNotConfigured(RuntimeError):
    pass


class GeminiExtractorError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiExtractionResult:
    language: str
    title_suggested: str
    summary: str
    keywords: list[str]
    raw_text: str
    sections: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    fields: dict[str, Any]
    confidence: float


def extract_document_with_gemini(
    *,
    file_path: Path,
    mime_type: str,
    original_filename: str | None,
    paperless_text: str = "",
) -> GeminiExtractionResult:
    """Use Gemini Flash to OCR and extract a scanned/image document."""
    api_key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not api_key:
        raise GeminiExtractorNotConfigured(
            f"{GEMINI_API_KEY_ENV} is not configured",
        )

    if not file_path.is_file():
        raise GeminiExtractorError(f"Source file does not exist: {file_path}")

    max_inline_bytes = _get_max_inline_bytes()
    file_bytes = file_path.read_bytes()
    if len(file_bytes) > max_inline_bytes:
        raise GeminiExtractorError(
            f"File is too large for inline Gemini upload: {len(file_bytes)} bytes",
        )

    model = os.getenv(GEMINI_MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL
    endpoint = os.getenv(GEMINI_ENDPOINT_ENV, DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT
    url = f"{endpoint.rstrip('/')}/models/{model}:generateContent"

    response_json = _generate_content(
        url=url,
        api_key=api_key,
        mime_type=mime_type,
        file_bytes=file_bytes,
        original_filename=original_filename,
        paperless_text=paperless_text,
    )
    text = _extract_response_text(response_json)
    parsed = _parse_json_object(text)

    return GeminiExtractionResult(
        language=str(parsed.get("language") or "unknown"),
        title_suggested=str(parsed.get("title_suggested") or original_filename or ""),
        summary=str(parsed.get("summary") or ""),
        keywords=_normalize_str_list(parsed.get("keywords")),
        raw_text=str(parsed.get("raw_text") or paperless_text or ""),
        sections=_normalize_dict_list(parsed.get("sections")),
        entities=_normalize_dict_list(parsed.get("entities")),
        fields=_normalize_dict(parsed.get("fields")),
        confidence=_normalize_confidence(parsed.get("confidence")),
    )


def _generate_content(
    *,
    url: str,
    api_key: str,
    mime_type: str,
    file_bytes: bytes,
    original_filename: str | None,
    paperless_text: str,
) -> dict[str, Any]:
    prompt = _build_prompt(original_filename=original_filename, paperless_text=paperless_text)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "application/octet-stream",
                            "data": base64.b64encode(file_bytes).decode("ascii"),
                        },
                    },
                ],
            },
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    }

    timeout_seconds = _get_timeout_seconds()
    timeout = httpx.Timeout(
        connect=30.0,
        read=timeout_seconds,
        write=60.0,
        pool=30.0,
    )
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        response = client.post(url, params={"key": api_key}, json=payload)
        response.raise_for_status()
        return response.json()


def _build_prompt(*, original_filename: str | None, paperless_text: str) -> str:
    paperless_hint = ""
    if paperless_text.strip():
        paperless_hint = (
            "\nPaperless OCR text is available but may be noisy. Use it only as a hint "
            "and correct it from the attached file:\n"
            f"{paperless_text[:6000]}\n"
        )

    return f"""
You are an OCR and document information extraction engine for an internal document pipeline.

File name: {original_filename or "unknown"}
{paperless_hint}
Read the attached document/image carefully. Return only one valid JSON object with this schema:
{{
  "language": "detected language code or name",
  "title_suggested": "short title",
  "summary": "short factual summary",
  "keywords": ["keyword"],
  "raw_text": "full OCR text in reading order",
  "sections": [{{"title": "section title", "text": "section text", "page": null}}],
  "entities": [{{"type": "person|organization|date|code|location|other", "text": "entity text", "confidence": 0.0}}],
  "fields": {{"key": "value"}},
  "confidence": 0.0
}}

Rules:
- Do not invent information.
- Preserve Vietnamese accents when visible.
- If a field is unknown, use an empty string, empty list, empty object, or null.
- Confidence must be a number from 0 to 1.
""".strip()


def _extract_response_text(response_json: dict[str, Any]) -> str:
    try:
        parts = response_json["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiExtractorError("Gemini response does not contain text parts") from exc

    texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
    text = "\n".join(t for t in texts if t.strip()).strip()
    if not text:
        raise GeminiExtractorError("Gemini response text is empty")
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise GeminiExtractorError("Gemini JSON response is not an object")
    return parsed


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.75
    return max(0.0, min(confidence, 1.0))


def _get_max_inline_bytes() -> int:
    raw = os.getenv(GEMINI_MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_INLINE_BYTES
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%s; using default", GEMINI_MAX_BYTES_ENV, raw)
        return DEFAULT_MAX_INLINE_BYTES


def _get_timeout_seconds() -> float:
    raw = os.getenv(GEMINI_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%s; using default", GEMINI_TIMEOUT_ENV, raw)
        return DEFAULT_TIMEOUT_SECONDS
    return max(30.0, timeout)
