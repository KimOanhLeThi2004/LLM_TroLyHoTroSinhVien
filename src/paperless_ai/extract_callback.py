import logging
import os

from celery import shared_task
from django.db import transaction
from rest_framework import serializers

from documents.data_models import DocumentMetadataOverrides
from documents.models import Document
from documents.plugins.helpers import DocumentsStatusManager
from paperless_ai.ai_status import STATUS_COMPLETED
from paperless_ai.ai_status import STATUS_FAILED
from paperless_ai.ai_status import STATUS_GEMINI_PROCESSING
from paperless_ai.ai_status import STATUS_PROCESSING
from paperless_ai.ai_status import STATUS_QUEUED
from paperless_ai.ai_status import set_ai_extract_status
from paperless_ai.digital_api_client import callback_extract_result
from paperless_ai.extract_schema import build_extract_schema
from paperless_ai.gemini_document_extractor import GeminiExtractorNotConfigured
from paperless_ai.gemini_document_extractor import extract_document_with_gemini
from paperless_ai.ocr_quality_check import check_ocr_quality

logger = logging.getLogger("paperless_ai.extract_callback")

WRITE_GEMINI_CONTENT_ENV = "PAPERLESS_AI_WRITE_GEMINI_CONTENT"


class GeminiFallbackRequiredButFailed(RuntimeError):
    pass


def handle_document_consumed(
    sender,
    document: Document,
    logging_group=None,
    original_file=None,
    ai_classification=None,
    **kwargs,
) -> None:
    """Queue AI extraction callback after Paperless has consumed the document."""
    content_length = len(document.content or "")
    classification = ai_classification or {}

    logger.info(
        "AI HOOK: document consumed id=%s title=%s content_length=%s page_count=%s mime_type=%s detected_type=%s extract_mode=%s original_file=%s",
        document.id,
        document.title,
        content_length,
        document.page_count,
        document.mime_type,
        classification.get("detected_type"),
        classification.get("extract_mode"),
        original_file,
        extra={"group": logging_group},
    )

    queued_reason = "Paperless consumed document; AI extraction is queued."
    queued_extract_mode = classification.get("extract_mode") or ""

    document.ai_extract_status = STATUS_QUEUED
    document.ai_extract_reason = queued_reason
    document.ai_extract_mode = queued_extract_mode
    document.ai_extract_score = None
    document.ai_extract_error = None

    set_ai_extract_status(
        document.id,
        STATUS_QUEUED,
        reason=queued_reason,
        extract_mode=queued_extract_mode,
    )

    transaction.on_commit(
        lambda: process_ai_extract_result.delay(document.id, classification),
    )


@shared_task(
    retry_backoff=True,
    max_retries=3,
)
def process_ai_extract_result(
    document_id: int,
    classification: dict | None = None,
    *,
    require_gemini_success: bool = False,
) -> None:
    """Normalize Paperless/Gemini extraction result and callback Tvdt.DigitalApi."""
    run_ai_extract_result(
        document_id,
        classification,
        require_gemini_success=require_gemini_success,
    )


def run_ai_extract_result(
    document_id: int,
    classification: dict | None = None,
    *,
    require_gemini_success: bool = False,
) -> None:
    """Run OCR quality scoring and Gemini fallback before exposing final content."""
    document = Document.objects.get(id=document_id)
    classification = classification or {}
    detected_type = classification.get("detected_type") or "unknown"
    extract_mode = classification.get("extract_mode") or "paperless-native"
    raw_text = document.content or ""
    language = "unknown"
    title_suggested = None
    summary = ""
    keywords = []
    sections = []
    entities = []
    fields = {}
    processing_log = [
        "Paperless consumption finished",
        f"classification={detected_type}",
        f"extract_mode={extract_mode}",
    ]
    error_info = None

    set_ai_extract_status(
        document.id,
        STATUS_PROCESSING,
        reason="AI quality check is running.",
        extract_mode=extract_mode,
    )

    confidence = float(classification.get("confidence") or 0.75)
    quality = check_ocr_quality(
        text=raw_text,
        page_count=document.page_count,
    )
    quality_payload = quality.as_dict()
    processing_log.append(
        f"ocr_quality_score={quality.score}; action={quality.recommended_action}",
    )

    should_use_gemini = detected_type == "scan-image" or not quality.passed
    if should_use_gemini:
        fallback_reason = (
            "scan-image route"
            if detected_type == "scan-image"
            else "Paperless content failed OCR quality check"
        )
        processing_log.append(f"Gemini Flash extraction started: {fallback_reason}")
        set_ai_extract_status(
            document.id,
            STATUS_GEMINI_PROCESSING,
            reason=f"Gemini Flash extraction started: {fallback_reason}",
            extract_mode="gemini-flash",
            score=quality.score,
        )
        try:
            gemini_result = extract_document_with_gemini(
                file_path=document.source_path,
                mime_type=document.mime_type,
                original_filename=document.original_filename,
                paperless_text=raw_text,
            )
            raw_text = gemini_result.raw_text
            language = gemini_result.language
            title_suggested = gemini_result.title_suggested
            summary = gemini_result.summary
            keywords = gemini_result.keywords
            sections = gemini_result.sections
            entities = gemini_result.entities
            fields = gemini_result.fields
            confidence = gemini_result.confidence
            extract_mode = "gemini-flash"
            classification["extract_mode"] = extract_mode
            _save_gemini_content_to_document(document, raw_text, processing_log)
            processing_log.append("Gemini Flash extraction completed")
        except GeminiExtractorNotConfigured as exc:
            processing_log.append("Gemini Flash extraction skipped: API key missing")
            error_info = {
                "code": "GEMINI_EXTRACTOR_NOT_CONFIGURED",
                "message": str(exc),
            }
            set_ai_extract_status(
                document.id,
                STATUS_FAILED,
                reason="Gemini Flash extraction skipped: API key missing",
                extract_mode="gemini-flash",
                score=quality.score,
                error=error_info,
            )
            confidence = min(confidence, 0.2)
            if require_gemini_success:
                raise GeminiFallbackRequiredButFailed(str(exc)) from exc
            _notify_document_updated(document.id)
        except Exception as exc:
            logger.exception(
                "Gemini Flash extraction failed document_id=%s",
                document.id,
            )
            processing_log.append("Gemini Flash extraction failed")
            error_info = {
                "code": "GEMINI_EXTRACTOR_FAILED",
                "message": str(exc),
            }
            set_ai_extract_status(
                document.id,
                STATUS_FAILED,
                reason="Gemini Flash extraction failed",
                extract_mode="gemini-flash",
                score=quality.score,
                error=error_info,
            )
            confidence = min(confidence, 0.2)
            if require_gemini_success:
                raise GeminiFallbackRequiredButFailed(str(exc)) from exc
            _notify_document_updated(document.id)
    else:
        processing_log.append("Using Paperless native extracted content")

    schema = build_extract_schema(
        document=document,
        classification=classification,
        raw_text=raw_text,
        confidence=confidence,
        language=language,
        title_suggested=title_suggested,
        summary=summary,
        keywords=keywords,
        sections=sections,
        entities=entities,
        fields=fields,
        ocr_quality=quality_payload,
        processing_log=processing_log,
        error_info=error_info,
    )
    payload = schema.as_dict()

    if error_info is None:
        set_ai_extract_status(
            document.id,
            STATUS_COMPLETED,
            reason="AI extraction completed.",
            extract_mode=extract_mode,
            score=quality.score,
        )
        _notify_document_updated(document.id)

    logger.info(
        "AI TASK: extract schema ready id=%s detected_type=%s extract_mode=%s raw_text_length=%s",
        document.id,
        detected_type,
        extract_mode,
        len(raw_text),
    )

    try:
        callback_extract_result(payload)
    except Exception:
        logger.exception(
            "DigitalApi extract callback failed document_id=%s",
            document.id,
        )


def _save_gemini_content_to_document(
    document: Document,
    raw_text: str,
    processing_log: list[str],
) -> None:
    """Persist Gemini OCR text into Paperless content so the UI/search can use it."""
    if not _should_write_gemini_content():
        processing_log.append("Gemini content DB update skipped by configuration")
        return

    normalized_text = raw_text.strip()
    if not normalized_text:
        processing_log.append("Gemini content DB update skipped: raw_text is empty")
        return

    if document.content == normalized_text:
        processing_log.append("Gemini content already matches Paperless content")
        return

    document.content = normalized_text
    document.save(update_fields=["content"])
    _update_search_index(document)
    processing_log.append("Gemini raw_text saved to Paperless document.content")


def _should_write_gemini_content() -> bool:
    value = os.getenv(WRITE_GEMINI_CONTENT_ENV, "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _update_search_index(document: Document) -> None:
    try:
        from documents.search import get_backend

        document.refresh_from_db()
        get_backend().add_or_update(
            document,
            effective_content=document.get_effective_content(),
        )
    except Exception:
        logger.exception(
            "Failed to update search index after Gemini content update document_id=%s",
            document.id,
        )


def _notify_document_updated(document_id: int) -> None:
    document = Document.objects.get(id=document_id)
    doc_overrides = DocumentMetadataOverrides.from_document(document)
    modified = serializers.DateTimeField().to_representation(document.modified)

    with DocumentsStatusManager() as status_mgr:
        status_mgr.send_document_updated(
            document_id=document.id,
            modified=modified,
            owner_id=doc_overrides.owner_id,
            users_can_view=doc_overrides.view_users,
            groups_can_view=doc_overrides.view_groups,
        )
