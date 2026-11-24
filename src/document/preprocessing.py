"""Preprocessing: render PDF pages and normalize images entirely in memory.

Outputs are consistent RGB PNG byte payloads suitable for a future vision
model. No OCR, AI call, or disk write happens here.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass

import pymupdf
from PIL import Image, ImageOps

from ..config import Settings
from ..schemas.document_ingestion import (
    DocumentFormat,
    PreprocessingResult,
    ProcessedPage,
)
from ..schemas.extraction_result import ProcessingStatus
from ..utils.logging_config import get_logger
from .ingestion import DocumentRecord, compute_sha256

logger = get_logger("document.preprocessing")


class PreprocessingError(ValueError):
    """Raised when a page cannot be rendered or an image cannot be normalized."""


@dataclass(frozen=True)
class NormalizedImage:
    """In-memory result of normalizing one rendered page or source image."""

    data: bytes
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    warnings: tuple[str, ...]
    exif_applied: bool = False
    converted_to_rgb: bool = False


def _image_to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def normalize_image(data: bytes, settings: Settings) -> NormalizedImage:
    """Correct EXIF orientation, flatten transparency, resize, and encode PNG."""
    warnings: list[str] = []
    try:
        with Image.open(io.BytesIO(data)) as source:
            original_width, original_height = source.size
            if original_width <= 0 or original_height <= 0:
                raise PreprocessingError("Image has invalid dimensions.")
            if original_width * original_height > settings.max_image_pixels:
                raise PreprocessingError(
                    "Image exceeds the configured pixel safety limit."
                )

            exif_orientation = None
            try:
                exif_orientation = source.getexif().get(274)
            except Exception:
                exif_orientation = None
            exif_applied = bool(
                settings.preserve_exif_orientation
                and exif_orientation not in (None, 1)
            )

            image = (
                ImageOps.exif_transpose(source)
                if settings.preserve_exif_orientation
                else source.copy()
            )
            converted_to_rgb = False
            try:
                source_mode = image.mode
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    flattened = Image.new("RGB", rgba.size, (255, 255, 255))
                    flattened.paste(rgba, mask=rgba.getchannel("A"))
                    rgba.close()
                    image.close()
                    image = flattened
                    converted_to_rgb = True
                    warnings.append(
                        f"Converted image mode {source_mode} to RGB and flattened transparency."
                    )
                elif image.mode != "RGB":
                    converted = image.convert("RGB")
                    image.close()
                    image = converted
                    converted_to_rgb = True
                    warnings.append(f"Converted image mode {source_mode} to RGB.")

                if exif_applied:
                    warnings.append("EXIF orientation was applied.")

                if max(image.size) > settings.vision_max_dimension:
                    image.thumbnail(
                        (settings.vision_max_dimension, settings.vision_max_dimension),
                        Image.Resampling.LANCZOS,
                    )
                    warnings.append(
                        f"Image resized to fit the {settings.vision_max_dimension}px limit."
                    )

                encoded = _image_to_png(image)
                return NormalizedImage(
                    data=encoded,
                    original_width=original_width,
                    original_height=original_height,
                    processed_width=image.width,
                    processed_height=image.height,
                    warnings=tuple(warnings),
                    exif_applied=exif_applied,
                    converted_to_rgb=converted_to_rgb,
                )
            finally:
                image.close()
    except PreprocessingError:
        raise
    except (OSError, ValueError) as exc:
        raise PreprocessingError(
            f"Image could not be normalized ({type(exc).__name__})."
        ) from exc


def render_pdf_page(
    pdf_bytes: bytes,
    page_number: int,
    settings: Settings,
) -> NormalizedImage:
    """Render one PDF page to a bounded-size RGB PNG in memory."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PreprocessingError(
            f"PDF could not be opened for rendering ({type(exc).__name__})."
        ) from exc

    try:
        if page_number < 1 or page_number > document.page_count:
            raise PreprocessingError(
                f"Page {page_number} is outside the 1-{document.page_count} range."
            )
        page = document.load_page(page_number - 1)
        rect = page.rect
        if rect.width <= 0 or rect.height <= 0:
            raise PreprocessingError(f"Page {page_number} has invalid dimensions.")

        warnings: list[str] = []
        scale = settings.pdf_render_dpi / 72.0
        estimated_pixels = (rect.width * scale) * (rect.height * scale)
        if estimated_pixels > settings.max_pdf_render_pixels:
            reduction = math.sqrt(settings.max_pdf_render_pixels / estimated_pixels)
            scale *= reduction
            warnings.append("Render scale reduced to respect the pixel budget.")

        try:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        except Exception as exc:
            raise PreprocessingError(
                f"Page {page_number} could not be rendered ({type(exc).__name__})."
            ) from exc

        rendered_width, rendered_height = pixmap.width, pixmap.height
        rendered_png = pixmap.tobytes("png")
        del pixmap
    finally:
        document.close()

    normalized = normalize_image(rendered_png, settings)
    return NormalizedImage(
        data=normalized.data,
        original_width=rendered_width,
        original_height=rendered_height,
        processed_width=normalized.processed_width,
        processed_height=normalized.processed_height,
        warnings=tuple(warnings) + normalized.warnings,
    )


def _select_pages(
    record: DocumentRecord,
    settings: Settings,
    page_numbers: list[int] | tuple[int, ...] | None,
) -> tuple[list[int], list[str]]:
    inspection = record.metadata.pdf
    if inspection is None:
        raise PreprocessingError("PDF metadata is missing; inspect the document first.")
    total = inspection.page_count
    warnings: list[str] = []

    if page_numbers is None:
        count = min(total, settings.max_pages_to_process)
        if total > count:
            warnings.append(
                f"Only the first {count} of {total} pages were processed by default."
            )
        return list(range(1, count + 1)), warnings

    valid: list[int] = []
    invalid: list[int] = []
    for requested in page_numbers:
        try:
            number = int(requested)
        except (TypeError, ValueError):
            invalid.append(-1)
            continue
        if 1 <= number <= total and number not in valid:
            valid.append(number)
        else:
            invalid.append(number)
    if invalid:
        warnings.append(f"Ignored invalid page selections: {invalid}.")
    valid.sort()
    if len(valid) > settings.max_pages_to_process:
        warnings.append(
            f"Page selection truncated to {settings.max_pages_to_process} pages."
        )
        valid = valid[: settings.max_pages_to_process]
    return valid, warnings


def _failed_page(
    record: DocumentRecord,
    page_number: int,
    error: str,
    elapsed: float,
) -> ProcessedPage:
    return ProcessedPage(
        document_id=record.document_id,
        original_filename=record.sanitized_filename,
        document_format=record.document_format,
        page_number=page_number,
        original_width=0,
        original_height=0,
        processed_width=0,
        processed_height=0,
        processing_status=ProcessingStatus.FAILED,
        processing_time_seconds=elapsed,
        sha256=compute_sha256(b""),
        image_bytes=b"",
        error=error[:2000],
    )


def _success_page(
    record: DocumentRecord,
    page_number: int,
    normalized: NormalizedImage,
    elapsed: float,
) -> ProcessedPage:
    return ProcessedPage(
        document_id=record.document_id,
        original_filename=record.sanitized_filename,
        document_format=record.document_format,
        page_number=page_number,
        original_width=normalized.original_width,
        original_height=normalized.original_height,
        processed_width=normalized.processed_width,
        processed_height=normalized.processed_height,
        processing_status=ProcessingStatus.COMPLETED,
        warnings=normalized.warnings,
        processing_time_seconds=elapsed,
        sha256=compute_sha256(normalized.data),
        image_bytes=normalized.data,
    )


def preprocess_document(
    record: DocumentRecord,
    settings: Settings,
    page_numbers: list[int] | tuple[int, ...] | None = None,
) -> PreprocessingResult:
    """Produce normalized in-memory page images for one validated document."""
    started = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []
    pages: list[ProcessedPage] = []

    try:
        if record.document_format is DocumentFormat.PDF:
            selected, selection_warnings = _select_pages(record, settings, page_numbers)
            warnings.extend(selection_warnings)
            for page_number in selected:
                page_started = time.perf_counter()
                try:
                    normalized = render_pdf_page(record.raw_bytes, page_number, settings)
                except PreprocessingError as exc:
                    message = f"Page {page_number}: {exc}"
                    errors.append(message)
                    pages.append(
                        _failed_page(
                            record, page_number, message, time.perf_counter() - page_started
                        )
                    )
                else:
                    pages.append(
                        _success_page(
                            record,
                            page_number,
                            normalized,
                            time.perf_counter() - page_started,
                        )
                    )
        else:
            page_started = time.perf_counter()
            try:
                normalized = normalize_image(record.raw_bytes, settings)
            except PreprocessingError as exc:
                message = f"Image: {exc}"
                errors.append(message)
                pages.append(
                    _failed_page(record, 1, message, time.perf_counter() - page_started)
                )
            else:
                pages.append(
                    _success_page(
                        record, 1, normalized, time.perf_counter() - page_started
                    )
                )
    except PreprocessingError as exc:
        errors.append(str(exc))

    successful = sum(
        1 for page in pages if page.processing_status is ProcessingStatus.COMPLETED
    )
    if successful == len(pages) and successful > 0:
        status = ProcessingStatus.COMPLETED
    elif successful > 0:
        status = ProcessingStatus.REQUIRES_REVIEW
        warnings.append("Some pages failed; successful pages are still available.")
    else:
        status = ProcessingStatus.FAILED

    result = PreprocessingResult(
        document_id=record.document_id,
        original_filename=record.sanitized_filename,
        document_format=record.document_format,
        status=status,
        pages=tuple(pages),
        warnings=tuple(warnings),
        errors=tuple(errors),
        processing_time_seconds=time.perf_counter() - started,
    )
    logger.info(
        "Preprocessed %s page(s) for document %s with status %s.",
        len(pages),
        record.document_id,
        status.value,
    )
    return result


def make_preview_from_bytes(data: bytes, max_dimension: int) -> bytes:
    """Create a display-sized PNG preview from in-memory image bytes."""
    if not data:
        return b""
    try:
        with Image.open(io.BytesIO(data)) as image:
            preview = image.copy()
            try:
                if max(preview.size) > max_dimension:
                    preview.thumbnail(
                        (max_dimension, max_dimension), Image.Resampling.LANCZOS
                    )
                return _image_to_png(preview)
            finally:
                preview.close()
    except (OSError, ValueError) as exc:
        raise PreprocessingError(
            f"Preview could not be generated ({type(exc).__name__})."
        ) from exc


def make_preview(page: ProcessedPage, max_dimension: int) -> bytes:
    """Create a display-sized preview without modifying the processed page."""
    return make_preview_from_bytes(page.image_bytes, max_dimension)


def clear_processed_bytes(result: PreprocessingResult) -> PreprocessingResult:
    """Return a copy without image payloads to release session memory."""
    return result.model_copy(
        update={
            "pages": tuple(
                page.model_copy(update={"image_bytes": b""}) for page in result.pages
            )
        }
    )
