from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

from documents.models import Document


@dataclass
class ExtractSchema:
    document_id: str
    paperless_document_id: int
    language: str
    detected_type: str
    title_suggested: str
    summary: str
    keywords: list[str]
    raw_text: str
    sections: list[dict]
    entities: list[dict]
    fields: dict
    confidence: float
    ocr_quality: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    extract_mode: str = "paperless-native"
    page_count: int | None = None
    file_name: str | None = None
    mime_type: str | None = None
    source_path: str | None = None
    processing_log: list[str] = field(default_factory=list)
    error_info: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def build_extract_schema(
    *,
    document: Document,
    classification: dict | None,
    raw_text: str,
    confidence: float,
    language: str = "unknown",
    title_suggested: str | None = None,
    summary: str = "",
    keywords: list[str] | None = None,
    sections: list[dict] | None = None,
    entities: list[dict] | None = None,
    fields: dict | None = None,
    ocr_quality: dict | None = None,
    processing_log: list[str] | None = None,
    error_info: dict | None = None,
) -> ExtractSchema:
    classification = classification or {}
    detected_type = classification.get("detected_type") or "unknown"
    extract_mode = classification.get("extract_mode") or "paperless-native"

    return ExtractSchema(
        document_id=str(document.id),
        paperless_document_id=document.id,
        language=language or "unknown",
        detected_type=detected_type,
        title_suggested=title_suggested or document.title or document.original_filename or "",
        summary=summary or "",
        keywords=keywords or [],
        raw_text=raw_text or "",
        sections=sections or [],
        entities=entities or [],
        fields=fields or {},
        confidence=confidence,
        ocr_quality=ocr_quality or {},
        classification=classification,
        extract_mode=extract_mode,
        page_count=document.page_count,
        file_name=document.original_filename,
        mime_type=document.mime_type,
        source_path=str(document.source_path) if document.source_path else None,
        processing_log=processing_log or [],
        error_info=error_info,
    )
