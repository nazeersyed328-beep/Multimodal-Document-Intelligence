"""Tests for in-memory document ingestion and validation."""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

from src.document.ingestion import (
    DocumentRecord,
    ingest_bytes,
    ingest_upload,
    ingest_uploads,
    sanitize_filename,
    validate_upload_bytes,
)
from src.schemas.document_ingestion import DocumentFormat, IngestionStatus

from .conftest import (
    FakeUpload,
    make_corrupt_pdf,
    make_corrupt_png,
    make_jpeg,
    make_pdf,
    make_png,
    make_test_settings,
    make_webp,
)


def ingest(name: str, data: bytes, settings, file_id: str | None = None):
    return ingest_upload(FakeUpload(name, data, file_id), settings)


def test_valid_png_upload(settings) -> None:
    data = make_png(40, 30)
    result, record = ingest("photo.png", data, settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.document_format is DocumentFormat.PNG
    assert record.metadata.image is not None
    assert (record.metadata.image.width, record.metadata.image.height) == (40, 30)
    assert record.raw_bytes == data
    assert record.fingerprint == result.metadata.sha256


def test_valid_jpeg_upload(settings) -> None:
    result, record = ingest("photo.jpeg", make_jpeg(), settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.document_format is DocumentFormat.JPEG
    assert record.metadata.image is not None
    assert record.metadata.image.image_format == "JPEG"


def test_valid_webp_upload(settings) -> None:
    result, record = ingest("diagram.webp", make_webp(), settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.document_format is DocumentFormat.WEBP


def test_valid_pdf_upload(settings) -> None:
    result, record = ingest("invoice.pdf", make_pdf(pages=2, with_text=True), settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.document_format is DocumentFormat.PDF
    assert record.metadata.pdf is not None
    assert record.metadata.pdf.page_count == 2
    assert [page.page_number for page in record.metadata.pdf.pages] == [1, 2]


def test_unsupported_file_rejected(settings) -> None:
    result, record = ingest("script.exe", b"MZ\x00\x00binary", settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None
    assert any("Unsupported file extension" in issue.message for issue in result.errors)


def test_empty_file_rejected(settings) -> None:
    result, record = ingest("empty.png", b"", settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None
    assert any("empty" in issue.message.lower() for issue in result.errors)


def test_corrupted_pdf_rejected(settings) -> None:
    result, record = ingest("broken.pdf", make_corrupt_pdf(), settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None
    assert any("PDF" in issue.message for issue in result.errors)


def test_corrupted_image_rejected(settings) -> None:
    result, record = ingest("broken.png", make_corrupt_png(), settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None
    assert any("corrupted" in issue.message.lower() for issue in result.errors)


def test_oversized_document_rejected() -> None:
    settings = make_test_settings(MAX_UPLOAD_SIZE_MB="1")
    oversized = b"\x89PNG\r\n\x1a\n" + os.urandom(1024 * 1024 + 64)
    result, record = ingest("huge.png", oversized, settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None
    assert any("exceeds" in issue.message for issue in result.errors)


def test_filename_sanitization(settings) -> None:
    assert sanitize_filename("../../etc/passwd?.png") == "passwd_.png"
    assert sanitize_filename("") == "document"
    assert sanitize_filename("..\\secret.png") == "secret.png"
    assert "/" not in sanitize_filename("a/b/c.png")
    assert "\\" not in sanitize_filename("a\\b\\c.png")

    result, record = ingest("../../reports/../safe name.png", make_png(), settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert "/" not in record.sanitized_filename
    assert "\\" not in record.sanitized_filename
    assert any("sanitized" in warning.message for warning in result.warnings)


def test_extension_cannot_override_signature(settings) -> None:
    result, record = ingest("fake.pdf", make_png(), settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None


def test_pdf_page_count_and_empty_page_detection(settings) -> None:
    result, record = ingest("blank.pdf", make_pdf(pages=3, with_text=False), settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.metadata.pdf is not None
    assert record.metadata.pdf.page_count == 3
    assert [page.page_number for page in record.metadata.pdf.pages] == [1, 2, 3]
    assert all(page.content_status.value == "empty" for page in record.metadata.pdf.pages)
    assert any("empty" in warning.message.lower() for warning in result.warnings)


def test_image_only_pdf_page_detection(settings) -> None:
    result, record = ingest(
        "scanned.pdf", make_pdf(pages=1, with_text=False, with_image=True), settings
    )
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.metadata.pdf is not None
    assert record.metadata.pdf.pages[0].content_status.value == "image_only"


def test_pdf_page_limit_rejection() -> None:
    settings = make_test_settings(MAX_PAGES_PER_PDF="2")
    result, record = ingest("long.pdf", make_pdf(pages=4), settings)
    assert result.status is IngestionStatus.REJECTED
    assert record is None
    assert any("limit" in issue.message for issue in result.errors)


def test_multiple_document_ingestion(settings) -> None:
    uploads = [
        FakeUpload("a.png", make_png(color=(1, 2, 3)), "a"),
        FakeUpload("b.jpg", make_jpeg(), "b"),
        FakeUpload("c.pdf", make_pdf(), "c"),
    ]
    batch = ingest_uploads(uploads, settings)
    assert len(batch.records) == 3
    assert batch.rejected == ()
    assert {record.document_format for record in batch.records} == {
        DocumentFormat.PNG,
        DocumentFormat.JPEG,
        DocumentFormat.PDF,
    }


def test_per_document_error_handling(settings) -> None:
    class BrokenUpload:
        name = "broken.png"

        def getvalue(self) -> bytes:
            raise RuntimeError("unreadable")

    batch = ingest_uploads(
        [BrokenUpload(), FakeUpload("good.png", make_png(), "good")],
        settings,
    )
    assert len(batch.records) == 1
    assert len(batch.rejected) == 1
    assert any("could not be read" in issue.message.lower() for issue in batch.rejected[0].errors)


def test_duplicate_detection(settings) -> None:
    data = make_png()
    batch = ingest_uploads(
        [
            FakeUpload("first.png", data, "1"),
            FakeUpload("copy.png", data, "2"),
        ],
        settings,
    )
    assert len(batch.records) == 1
    assert batch.duplicates == ("copy.png",)


def test_session_document_limit() -> None:
    settings = make_test_settings(MAX_DOCUMENTS_PER_SESSION="2")
    uploads = [
        FakeUpload(f"color-{index}.png", make_png(color=(index, 1, 2)), str(index))
        for index in range(4)
    ]
    batch = ingest_uploads(uploads, settings)
    assert len(batch.records) == 2
    assert len(batch.skipped_limit) == 2


def test_validate_upload_bytes_returns_structured_result(settings) -> None:
    validation, document_format = validate_upload_bytes("x.webp", make_webp(), settings)
    assert validation.is_valid is True
    assert validation.extension == ".webp"
    assert document_format is DocumentFormat.WEBP


def test_metadata_excludes_raw_bytes(settings) -> None:
    result, record = ingest("clean.png", make_png(), settings)
    assert record is not None
    serialized = str(result.metadata.model_dump())
    assert "raw_bytes" not in serialized
    assert "\\x89PNG" not in serialized


def test_rgba_metadata(settings) -> None:
    image = Image.new("RGBA", (12, 8), (255, 0, 0, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result, record = ingest("transparent.png", buffer.getvalue(), settings)
    assert result.status is IngestionStatus.VALIDATED
    assert record is not None
    assert record.metadata.image is not None
    assert record.metadata.image.has_transparency is True


@pytest.mark.parametrize("format_name", ["png", "jpeg", "webp"])
def test_supported_image_formats_are_accepted(settings, format_name: str) -> None:
    data = {
        "png": make_png(),
        "jpeg": make_jpeg(),
        "webp": make_webp(),
    }[format_name]
    result, record = ingest(f"sample.{format_name}", data, settings)
    assert result.status is IngestionStatus.VALIDATED
    assert isinstance(record, DocumentRecord)
