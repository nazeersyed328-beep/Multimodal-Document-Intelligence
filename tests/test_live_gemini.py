"""Optional live Gemini smoke test.

Skipped by default. To run with a real key:
``RUN_LIVE_GEMINI_TESTS=1 GEMINI_API_KEY=... py -m pytest tests/test_live_gemini.py``.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pytest
from PIL import Image, ImageDraw

from src.config import load_settings
from src.document.ingestion import ingest_upload
from src.document.preprocessing import preprocess_document
from src.extraction import ExtractionService
from src.schemas.common import DocumentType
from src.schemas.extraction_result import ProcessingStatus

from .conftest import FakeUpload

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI_TESTS") != "1",
    reason="Set RUN_LIVE_GEMINI_TESTS=1 to enable real API calls.",
)


def _invoice_image() -> bytes:
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), "INVOICE", fill="black")
    draw.text((40, 100), "Vendor: Acme Supplies", fill="black")
    draw.text((40, 150), "Invoice Number: INV-LIVE-1", fill="black")
    draw.text((40, 200), "Invoice Date: 2026-01-02", fill="black")
    draw.text((40, 250), "Currency: USD", fill="black")
    draw.text((40, 300), "Subtotal: 10.00", fill="black")
    draw.text((40, 350), "Tax: 1.00", fill="black")
    draw.text((40, 400), "Total: 11.00", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_live_gemini_extraction_smoke() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.lower().startswith("your-"):
        pytest.skip("GEMINI_API_KEY is not configured for live testing.")

    settings = load_settings(environ={"GEMINI_API_KEY": api_key})
    _, record = ingest_upload(
        FakeUpload("live-invoice.png", _invoice_image(), "image/png"),
        settings,
    )
    assert record is not None
    processed = preprocess_document(record, settings)

    result = ExtractionService(settings).extract_document(
        record, processed, DocumentType.INVOICE
    )

    assert result.status is not ProcessingStatus.FAILED, result.validation_errors
    assert result.data is not None
    assert result.processed_at is not None
    assert result.processed_at.tzinfo == timezone.utc
