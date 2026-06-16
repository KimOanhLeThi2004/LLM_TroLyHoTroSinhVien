from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from paperless.parsers.utils import PDF_TEXT_MIN_LENGTH
from paperless.parsers.utils import extract_pdf_text
from paperless.parsers.utils import get_page_count_for_pdf

MIN_NATIVE_PDF_TEXT_LENGTH = 1_000
MIN_NATIVE_PDF_CHARS_PER_PAGE = 100

TEXT_NATIVE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".html",
    ".htm",
    ".md",
    ".odt",
    ".rtf",
    ".txt",
    ".xml",
}

TEXT_NATIVE_MIME_PREFIXES = (
    "text/",
)

TEXT_NATIVE_MIME_TYPES = {
    "application/msword",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/xml",
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class DocumentClassification:
    detected_type: str
    extract_mode: str
    confidence: float
    reason: str
    mime_type: str
    extension: str
    parser_name: str | None = None
    pdf_text_length: int | None = None
    pdf_page_count: int | None = None
    pdf_chars_per_page: float | None = None
    threshold: int = MIN_NATIVE_PDF_TEXT_LENGTH

    def as_dict(self) -> dict:
        return asdict(self)


def classify_document_input(
    *,
    file_path: Path,
    file_name: str,
    mime_type: str,
    parser_name: str | None = None,
) -> DocumentClassification:
    """Classify using Paperless-detected MIME type and selected parser."""
    extension = Path(file_name).suffix.lower()

    if mime_type == "application/pdf" or extension == ".pdf":
        native_text = extract_pdf_text(file_path)
        text_length = len(native_text or "")
        page_count = get_page_count_for_pdf(file_path)
        chars_per_page = (
            text_length / page_count
            if page_count and page_count > 0
            else None
        )
        has_native_text_layer = text_length >= MIN_NATIVE_PDF_TEXT_LENGTH
        if chars_per_page is not None:
            has_native_text_layer = (
                has_native_text_layer
                and chars_per_page >= MIN_NATIVE_PDF_CHARS_PER_PAGE
            )

        if has_native_text_layer:
            return DocumentClassification(
                detected_type="text-native",
                extract_mode="paperless-native",
                confidence=0.9,
                reason="PDF has a useful text layer above total/per-page thresholds",
                mime_type=mime_type,
                extension=extension,
                parser_name=parser_name,
                pdf_text_length=text_length,
                pdf_page_count=page_count,
                pdf_chars_per_page=chars_per_page,
            )

        return DocumentClassification(
            detected_type="scan-image",
            extract_mode="gemini-flash",
            confidence=0.9,
            reason="PDF text layer is missing, too short, or too sparse per page",
            mime_type=mime_type,
            extension=extension,
            parser_name=parser_name,
            pdf_text_length=text_length,
            pdf_page_count=page_count,
            pdf_chars_per_page=chars_per_page,
        )

    if mime_type.startswith("image/") or extension in IMAGE_EXTENSIONS:
        return DocumentClassification(
            detected_type="scan-image",
            extract_mode="gemini-flash",
            confidence=0.95,
            reason="Image document requires OCR",
            mime_type=mime_type,
            extension=extension,
            parser_name=parser_name,
        )

    if (
        mime_type in TEXT_NATIVE_MIME_TYPES
        or mime_type.startswith(TEXT_NATIVE_MIME_PREFIXES)
        or extension in TEXT_NATIVE_EXTENSIONS
    ):
        return DocumentClassification(
            detected_type="text-native",
            extract_mode="paperless-native",
            confidence=0.85,
            reason="Native text document type",
            mime_type=mime_type,
            extension=extension,
            parser_name=parser_name,
        )

    return DocumentClassification(
        detected_type="text-native",
        extract_mode="paperless-native",
        confidence=0.5,
        reason="Unknown type; falling back to Paperless parser",
        mime_type=mime_type,
        extension=extension,
        parser_name=parser_name,
    )
