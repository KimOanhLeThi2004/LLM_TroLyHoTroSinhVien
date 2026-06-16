from __future__ import annotations

from typing import Any

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_GEMINI_PROCESSING = "gemini_processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

DEFAULT_STATUS = STATUS_COMPLETED
_PROCESSING_STATUSES = {STATUS_QUEUED, STATUS_PROCESSING, STATUS_GEMINI_PROCESSING}
_VALID_STATUSES = _PROCESSING_STATUSES | {STATUS_COMPLETED, STATUS_FAILED}


def set_ai_extract_status(
    document_id: int,
    status: str,
    *,
    reason: str = "",
    extract_mode: str = "",
    score: float | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    from documents.models import Document

    normalized_status = status if status in _VALID_STATUSES else STATUS_PROCESSING
    Document.objects.filter(id=document_id).update(
        ai_extract_status=normalized_status,
        ai_extract_reason=reason,
        ai_extract_mode=extract_mode,
        ai_extract_score=score,
        ai_extract_error=error,
    )


def get_ai_extract_status(document_id: int) -> dict[str, Any]:
    from documents.models import Document

    payload = (
        Document.objects.filter(id=document_id)
        .values(
            "ai_extract_status",
            "ai_extract_reason",
            "ai_extract_mode",
            "ai_extract_score",
            "ai_extract_error",
        )
        .first()
    )
    if payload is None:
        return _default_status_payload()

    status = payload.get("ai_extract_status") or DEFAULT_STATUS
    return {
        "status": status,
        "reason": payload.get("ai_extract_reason") or "",
        "extract_mode": payload.get("ai_extract_mode") or "",
        "score": payload.get("ai_extract_score"),
        "error": payload.get("ai_extract_error"),
        "is_processing": status in _PROCESSING_STATUSES,
    }


def is_ai_extract_processing(document_id: int) -> bool:
    return bool(get_ai_extract_status(document_id).get("is_processing"))


def _default_status_payload() -> dict[str, Any]:
    return {
        "status": DEFAULT_STATUS,
        "reason": "No active AI extraction task.",
        "extract_mode": "",
        "score": None,
        "error": None,
        "is_processing": False,
    }
