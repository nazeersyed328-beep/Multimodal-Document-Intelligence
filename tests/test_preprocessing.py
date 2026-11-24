"""Tests for PDF rendering, image normalization, and preprocessing results."""

from __future__ import annotations

import io
from dataclasses import replace

from PIL import Image

from src.document.ingestion import ingest_upload
from src.document.preprocessing import (
    PreprocessingError,
    clear_processed_bytes,
    make_preview,
    make_preview_from_bytes,
    normalize_image,
    preprocess_document,
    render_pdf_page,
)
from src.schemas.document_ingestion import PreprocessingResult, ProcessedPage
from src.schemas.extraction_result import ProcessingStatus

from .conftest import (
    FakeUpload,
    make_corrupt_png,
    make_jpeg,
    make_pdf,
    make_png,
    make_test_settings,
)


def record_for(name: str, data: bytes, settings):
    result, record = ingest_upload(FakeUpload(name, data, name), settings)
    assert result.status.value == "validated"
    assert record is not None
    return record


def test_image_preprocessing_output_schema(settings) -> None:
    record = record_for("page.png", make_png(), settings)
    result = preprocess_document(record, settings)
    assert isinstance(result, PreprocessingResult)
    assert result.document_id == record.document_id
    assert result.document_format.value == "png"
    assert result.status is ProcessingStatus.COMPLETED
    assert len(result.pages) == 1
    page = result.pages[0]
    assert isinstance(page, ProcessedPage)
    assert page.image_format == "PNG"
    assert page.processing_status is ProcessingStatus.COMPLETED
    assert page.processed_width > 0 and page.processed_height > 0
    assert len(page.sha256) == 64
    assert page.processing_time_seconds >= 0


def test_rgba_converted_to_rgb(settings) -> None:
    data = make_png(30, 20, mode="RGBA", color=(255, 0, 0, 128))
    record = record_for("alpha.png", data, settings)
    result = preprocess_document(record, settings)
    page = result.pages[0]
    with Image.open(io.BytesIO(page.image_bytes)) as image:
        assert image.mode == "RGB"
    assert any("flattened" in warning for warning in page.warnings)


def test_palette_image_converted_to_rgb(settings) -> None:
    image = Image.new("P", (20, 10), 3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    record = record_for("palette.png", buffer.getvalue(), settings)
    result = preprocess_document(record, settings)
    with Image.open(io.BytesIO(result.pages[0].image_bytes)) as processed:
        assert processed.mode == "RGB"


def test_image_resizing_and_aspect_ratio(settings) -> None:
    record = record_for("wide.png", make_png(800, 400), settings)
    result = preprocess_document(record, settings)
    page = result.pages[0]
    assert (page.original_width, page.original_height) == (800, 400)
    assert (page.processed_width, page.processed_height) == (400, 200)
    assert abs(page.processed_width / page.processed_height - 2.0) < 0.01
    assert any("resized" in warning for warning in page.warnings)


def test_exif_orientation_applied(settings) -> None:
    data = make_jpeg(width=60, height=30, orientation=6)
    record = record_for("rotated.jpg", data, settings)
    assert record.metadata.image is not None
    assert record.metadata.image.exif_orientation == 6
    result = preprocess_document(record, settings)
    page = result.pages[0]
    assert (page.original_width, page.original_height) == (60, 30)
    assert (page.processed_width, page.processed_height) == (30, 60)
    assert any("EXIF" in warning for warning in page.warnings)


def test_pdf_page_rendering_uses_dpi(settings) -> None:
    data = make_pdf(pages=1, width=200, height=300)
    normalized = render_pdf_page(data, 1, settings)
    assert (normalized.original_width, normalized.original_height) == (200, 300)
    assert normalized.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_invalid_pdf_page_rejected(settings) -> None:
    data = make_pdf(pages=1)
    try:
        render_pdf_page(data, 4, settings)
    except PreprocessingError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Expected PreprocessingError")


def test_multipage_pdf_preprocessing(settings) -> None:
    record = record_for("multi.pdf", make_pdf(pages=3), settings)
    result = preprocess_document(record, settings)
    assert result.status is ProcessingStatus.COMPLETED
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert all(page.image_bytes for page in result.pages)


def test_pdf_page_selection(settings) -> None:
    record = record_for("multi.pdf", make_pdf(pages=3), settings)
    result = preprocess_document(record, settings, page_numbers=[2])
    assert [page.page_number for page in result.pages] == [2]


def test_pdf_default_page_cap_warning() -> None:
    settings = make_test_settings(MAX_PAGES_TO_PROCESS="2")
    record = record_for("five.pdf", make_pdf(pages=5), settings)
    result = preprocess_document(record, settings)
    assert [page.page_number for page in result.pages] == [1, 2]
    assert any("first 2" in warning for warning in result.warnings)


def test_pdf_pixel_budget_warning() -> None:
    settings = make_test_settings(MAX_PDF_RENDER_PIXELS="10000")
    normalized = render_pdf_page(make_pdf(pages=1), 1, settings)
    assert any("pixel budget" in warning for warning in normalized.warnings)


def test_failed_page_is_reported_not_ignored(settings) -> None:
    record = record_for("image.png", make_png(), settings)
    broken = replace(record, raw_bytes=make_corrupt_png())
    result = preprocess_document(broken, settings)
    assert result.status is ProcessingStatus.FAILED
    assert result.errors
    assert result.pages[0].processing_status is ProcessingStatus.FAILED
    assert result.pages[0].image_bytes == b""


def test_processed_bytes_excluded_from_serialization(settings) -> None:
    record = record_for("page.png", make_png(), settings)
    result = preprocess_document(record, settings)
    assert "image_bytes" not in result.pages[0].model_dump()
    assert "image_bytes" not in str(result.model_dump())


def test_preview_generation(settings) -> None:
    normalized = normalize_image(make_png(800, 400), settings)
    preview = make_preview_from_bytes(normalized.data, settings.preview_max_dimension)
    with Image.open(io.BytesIO(preview)) as image:
        assert max(image.size) <= settings.preview_max_dimension
        assert image.width / image.height == 2.0


def test_make_preview_from_page(settings) -> None:
    record = record_for("page.png", make_png(), settings)
    page = preprocess_document(record, settings).pages[0]
    preview = make_preview(page, settings.preview_max_dimension)
    assert preview.startswith(b"\x89PNG\r\n\x1a\n")


def test_clear_processed_bytes_releases_memory(settings) -> None:
    record = record_for("page.png", make_png(), settings)
    result = preprocess_document(record, settings)
    assert result.pages[0].image_bytes
    cleared = clear_processed_bytes(result)
    assert cleared.pages[0].image_bytes == b""
    assert cleared.status is result.status
    assert result.pages[0].image_bytes
