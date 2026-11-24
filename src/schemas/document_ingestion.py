"""Structured schemas for Phase 2 ingestion and preprocessing.

These models describe only local file handling. They contain no extracted
document facts, no confidence scores, and no AI output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import Field

from .common import DocumentId, ErrorSeverity, SourceFilename, StrictModel
from .extraction_result import ProcessingStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentFormat(str, Enum):
    """Formats accepted by the ingestion pipeline."""

    PDF = "pdf"
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class IngestionStatus(str, Enum):
    """Result of validating and inspecting an uploaded document."""

    VALIDATED = "validated"
    REJECTED = "rejected"


class PageContentStatus(str, Enum):
    """Coarse content classification derived from local PDF structure."""

    TEXT = "text"
    IMAGE_ONLY = "image_only"
    EMPTY = "empty"
    MIXED = "mixed"


class PageInfo(StrictModel):
    """Structural information about one PDF page."""

    page_number: int = Field(ge=1)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)
    rotation: int = Field(ge=0, le=360)
    content_status: PageContentStatus
    text_characters: int = Field(ge=0)
    image_count: int = Field(ge=0)


class PdfMetadata(StrictModel):
    """Optional embedded PDF metadata fields."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None


class PdfInspection(StrictModel):
    """Result of inspecting a PDF without rendering it."""

    page_count: int = Field(gt=0)
    is_encrypted: bool
    needs_password: bool
    metadata: PdfMetadata
    pages: tuple[PageInfo, ...]
    warnings: tuple[str, ...] = ()


class ImageInfo(StrictModel):
    """Basic properties read from an image header."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str = Field(min_length=1)
    image_format: str = Field(min_length=1)
    has_transparency: bool
    exif_orientation: int | None = Field(default=None, ge=1, le=8)


class DocumentMetadata(StrictModel):
    """Safe, non-sensitive metadata for one ingested document."""

    document_id: DocumentId
    original_filename: str = Field(min_length=1, max_length=255)
    sanitized_filename: SourceFilename
    document_format: DocumentFormat
    extension: str = Field(pattern=r"^\.[a-z0-9]+$")
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: ImageInfo | None = None
    pdf: PdfInspection | None = None


class IngestionIssue(StrictModel):
    """One validation or inspection problem for a document."""

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    severity: ErrorSeverity = ErrorSeverity.ERROR


class IngestionResult(StrictModel):
    """Outcome of ingesting one uploaded file."""

    document_id: DocumentId
    original_filename: str = Field(min_length=1, max_length=255)
    status: IngestionStatus
    metadata: DocumentMetadata | None = None
    errors: tuple[IngestionIssue, ...] = ()
    warnings: tuple[IngestionIssue, ...] = ()
    created_at: datetime = Field(default_factory=_utc_now)


class ProcessedPage(StrictModel):
    """One normalized in-memory image prepared for future vision input."""

    document_id: DocumentId
    original_filename: SourceFilename
    document_format: DocumentFormat
    page_number: int = Field(ge=1)
    original_width: int = Field(ge=0)
    original_height: int = Field(ge=0)
    processed_width: int = Field(ge=0)
    processed_height: int = Field(ge=0)
    image_format: str = "PNG"
    processing_status: ProcessingStatus
    warnings: tuple[str, ...] = ()
    processing_time_seconds: float = Field(ge=0, allow_inf_nan=False)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_bytes: bytes = Field(default=b"", repr=False, exclude=True)
    error: str | None = None


class PreprocessingResult(StrictModel):
    """All processed pages produced from one document."""

    document_id: DocumentId
    original_filename: SourceFilename
    document_format: DocumentFormat
    status: ProcessingStatus
    pages: tuple[ProcessedPage, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    processing_time_seconds: float = Field(ge=0, allow_inf_nan=False)
    created_at: datetime = Field(default_factory=_utc_now)
