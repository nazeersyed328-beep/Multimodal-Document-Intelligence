"""Synthetic file helpers shared by Phase 2 tests.

All fixtures are generated in memory. No real personal documents, network
access, API keys, or GPU are used.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pymupdf
import pytest
from PIL import Image

from src.config import Settings, load_settings
from src.schemas.common import ValidationStatus
from src.schemas.extraction_result import ExtractionResult, ProcessingStatus


@dataclass
class FakeUpload:
    """Minimal stand-in for a Streamlit UploadedFile."""

    name: str
    data: bytes
    file_id: str | None = None

    def getvalue(self) -> bytes:
        return self.data


def make_test_settings(**environment_overrides: str) -> Settings:
    environment = {
        "MAX_UPLOAD_SIZE_MB": "5",
        "MAX_PAGES_PER_PDF": "10",
        "MAX_PAGES_TO_PROCESS": "3",
        "PDF_RENDER_DPI": "72",
        "MAX_PDF_RENDER_PIXELS": "10000000",
        "MAX_IMAGE_PIXELS": "10000000",
        "PREVIEW_MAX_DIMENSION": "300",
        "VISION_MAX_DIMENSION": "400",
        "MAX_DOCUMENTS_PER_SESSION": "3",
        "LOG_LEVEL": "WARNING",
    }
    environment.update(environment_overrides)
    return load_settings(environ=environment)


def make_png(
    width: int = 40,
    height: int = 30,
    mode: str = "RGB",
    color: tuple[int, ...] = (120, 60, 200),
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_jpeg(
    width: int = 60,
    height: int = 30,
    orientation: int = 1,
) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (width, height), (200, 100, 50))
    if orientation == 1:
        image.save(buffer, format="JPEG")
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(buffer, format="JPEG", exif=exif.tobytes())
    return buffer.getvalue()


def make_webp(width: int = 35, height: int = 25) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 180, 90)).save(buffer, format="WEBP")
    return buffer.getvalue()


def make_pdf(
    pages: int = 1,
    *,
    with_text: bool = True,
    with_image: bool = False,
    width: int = 200,
    height: int = 300,
) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page(width=width, height=height)
        if with_text:
            page.insert_text((20, 30), f"Page {index + 1}")
        if with_image:
            page.insert_image(
                pymupdf.Rect(20, 50, 180, 250),
                stream=make_png(40, 30),
            )
    payload = document.tobytes()
    document.close()
    return payload


def make_corrupt_pdf() -> bytes:
    return b"%PDF-1.7\nthis is not a structurally valid pdf document\n"


def make_corrupt_png() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"corrupted image payload" * 4


class FakeVisionProvider:
    """Deterministic provider stand-in; performs no network calls."""

    name = "fake"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[Any] = []

    def extract(self, request: Any) -> Any:
        from src.extraction.provider import VisionResponse

        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeVisionProvider ran out of responses.")
        return VisionResponse(
            text=self.responses.pop(0),
            provider=self.name,
            model="fake-model",
        )


def make_extraction_settings(**environment_overrides: str) -> Settings:
    environment = {
        "GEMINI_API_KEY": "test-key-not-real",
        "GEMINI_MODEL": "gemini-3.8-flash",
        "TEMPERATURE": "0",
        "MAX_OUTPUT_TOKENS": "4096",
        "MAX_RETRIES": "2",
        "RETRY_BACKOFF_SECONDS": "0.01",
        "CONFIDENCE_REVIEW_THRESHOLD": "0.7",
    }
    environment.update(environment_overrides)
    return make_test_settings(**environment)


@pytest.fixture()
def settings() -> Settings:
    return make_test_settings()


@pytest.fixture()
def extraction_settings() -> Settings:
    return make_extraction_settings()


def make_result(
    data: Any | None,
    *,
    result_id: str = "result-test",
    source_document_id: str = "doc-test",
    status: ProcessingStatus = ProcessingStatus.COMPLETED,
    validation_status: ValidationStatus = ValidationStatus.VALID_WITH_WARNINGS,
    **overrides: Any,
) -> ExtractionResult:
    """Build a valid ExtractionResult for quality and review tests."""
    values: dict[str, Any] = {
        "result_id": result_id,
        "document_type": data.document_type if data is not None else overrides.pop("document_type"),
        "source_document_id": source_document_id,
        "status": status,
        "data": data,
        "validation_status": validation_status,
        "processed_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return ExtractionResult(**values)
