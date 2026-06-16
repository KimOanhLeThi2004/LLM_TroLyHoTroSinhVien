from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("paperless_ai.digital_api_client")

CALLBACK_URL_ENV = "PAPERLESS_DIGITAL_API_CALLBACK_URL"
CALLBACK_TOKEN_ENV = "PAPERLESS_DIGITAL_API_CALLBACK_TOKEN"


def callback_extract_result(payload: dict) -> None:
    """Send extract result to Tvdt.DigitalApi when callback URL is configured."""
    callback_url = os.getenv(CALLBACK_URL_ENV, "").strip()
    if not callback_url:
        logger.info(
            "DigitalApi callback URL is not configured; extract payload is ready: %s",
            payload,
        )
        return

    headers = {"Content-Type": "application/json"}
    token = os.getenv(CALLBACK_TOKEN_ENV, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=15.0, follow_redirects=False) as client:
        response = client.post(callback_url, json=payload, headers=headers)
        response.raise_for_status()

    logger.info(
        "DigitalApi extract callback sent paperless_document_id=%s url=%s",
        payload.get("paperless_document_id"),
        callback_url,
    )
