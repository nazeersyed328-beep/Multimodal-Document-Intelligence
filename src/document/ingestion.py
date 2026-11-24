"""Document ingestion: validate upload bytes, inspect structure, create records.

All work happens in memory and within the current Streamlit session. Nothing is
written to disk, no external service is contacted, and no extraction occurs.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf
from PIL import Image, UnidentifiedImageError

from ..config import Settings
from ..schemas.common import ErrorSeverity
from ..schemas.document_ingestion import (
    DocumentFormat,
    DocumentMetadata,
    ImageInfo,
    IngestionIssue,
    IngestionResult,
    IngestionStatus,
    PageContentStatus,
    PageInfo,
    PdfInspection,
    PdfMetadata,
)
from ..utils.logging_config import get_logger
from .validators import (
    DocumentValidationError,
    DocumentValidationResult,
    detect_mime_type,
    extract_extension,
    validate_extension,
    validate_file_size,
    validate_filename,
    validate_mime_type,
    validate_not_empty,
    validate_signature,
)

logger = get_logger("document.ingestion")

UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
MIME_TO_FORMAT = {
    "application/pdf": DocumentFormat.PDF,
    "image/png": DocumentFormat.PNG,
    "image/jpeg": DocumentFormat.JPEG,
    "image/webp": DocumentFormat.WEBP,
}
FORMAT_TO_MIME = {value: key for key, value in MIME_TO_FORMAT.items()}
MAX_METADATA_LENGTH = 500


class IngestionError(ValueError):
    """Raised when an uploaded document cannot be safely ingested."""


class UploadLike(Protocol):
    """Minimal interface shared by Streamlit uploads and in-memory test doubles."""

    name: str

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class DocumentRecord:
    """A validated document held only for the lifetime of the session."""

    document_id: str
    original_filename: str
    sanitized_filename: str
    document_format: DocumentFormat
    extension: str
    mime_type: str
    size_bytes: int
    sha256: str
    raw_bytes: bytes
    metadata: DocumentMetadata
    warnings: tuple[str, ...] = ()
    upload_id: str | None = None

    @property
    def fingerprint(self) -> str:
        return self.sha256


@dataclass(frozen=True)
class BatchIngestionResult:
    """Outcome for a group of independent uploads."""

    records: tuple[DocumentRecord, ...]
    rejected: tuple[IngestionResult, ...]
    duplicates: tuple[str, ...]
    skipped_limit: tuple[str, ...]


def new_document_id() -> str:
    """Return a unique identifier safe for use in schemas and UI keys."""
    return f"doc-{uuid.uuid4().hex}"


def compute_sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest used for duplicate detection."""
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Reduce a possibly hostile upload name to a safe bare filename."""
    base = filename.replace("\\", "/").split("/")[-1]
    base = unicodedata.normalize("NFKD", base)
    base = UNSAFE_FILENAME_CHARS.sub("_", base).strip(" .")
    if not base:
        base = "document"
    path = re.sub(r"_{2,}", "_", base)
    extension = re.sub(r"[^.a-z0-9]", "", Path(path).suffix.lower())
    stem = Path(path).stem
    if not stem or stem in {".", ".."}:
        stem = "document"
    if len(stem) > 100:
        stem = stem[:100]
    safe_name = f"{stem}{extension}" if extension else stem
    validate_filename(safe_name)
    return safe_name


def read_upload_bytes(upload: object) -> bytes:
    """Extract bytes from a Streamlit UploadedFile, BytesIO, or raw bytes."""
    if isinstance(upload, (bytes, bytearray, memoryview)):
        return bytes(upload)
    getvalue = getattr(upload, "getvalue", None)
    if callable(getvalue):
        return bytes(getvalue())
    getbuffer = getattr(upload, "getbuffer", None)
    if callable(getbuffer):
        return bytes(getbuffer())
    read = getattr(upload, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
    raise IngestionError("Uploaded file object does not expose readable bytes.")


def _upload_name(upload: object) -> str:
    name = getattr(upload, "name", "")
    return name if isinstance(name, str) else ""


def _upload_id(upload: object) -> str | None:
    value = getattr(upload, "file_id", None)
    return str(value) if value is not None else None


def detect_format(data: bytes, filename: str) -> DocumentFormat:
    """Detect format from signature bytes, then cross-check the extension."""
    mime = detect_mime_type(data[:16])
    if mime is None:
        raise IngestionError(
            "File signature does not match a supported PDF or image format."
        )
    detected = MIME_TO_FORMAT[mime]
    expected = {
        ".pdf": DocumentFormat.PDF,
        ".png": DocumentFormat.PNG,
        ".jpg": DocumentFormat.JPEG,
        ".jpeg": DocumentFormat.JPEG,
        ".webp": DocumentFormat.WEBP,
    }.get(extract_extension(filename))
    if expected is not None and expected is not detected:
        raise IngestionError(
            f"Extension indicates {expected.value} but file bytes indicate {detected.value}."
        )
    return detected


def validate_upload_bytes(
    filename: str,
    data: bytes,
    settings: Settings,
) -> tuple[DocumentValidationResult, DocumentFormat | None]:
    """Run Phase 1 structural checks against in-memory bytes (no temp files)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not filename or not filename.strip():
        errors.append("Filename must not be empty.")
    if len(filename) > 255:
        errors.append("Filename must not exceed 255 characters.")

    sanitized = sanitize_filename(filename) if filename else "document"
    if sanitized != filename:
        warnings.append("Filename was sanitized for safe handling.")

    try:
        validate_filename(sanitized)
    except DocumentValidationError as exc:
        errors.append(str(exc))

    try:
        extension = validate_extension(sanitized, settings.allowed_file_extensions)
    except DocumentValidationError as exc:
        errors.append(str(exc))
        extension = None

    try:
        validate_not_empty(len(data))
    except DocumentValidationError as exc:
        errors.append(str(exc))

    try:
        validate_file_size(len(data), settings.max_upload_size_bytes)
    except DocumentValidationError as exc:
        errors.append(str(exc))

    detected_format: DocumentFormat | None = None
    if extension is not None and data:
        header = data[:16]
        try:
            validate_signature(header, extension)
            validate_mime_type(sanitized, header)
            detected_format = detect_format(data, sanitized)
        except (DocumentValidationError, IngestionError) as exc:
            errors.append(str(exc))

    validation = DocumentValidationResult(
        filename=sanitized,
        extension=extension,
        size_bytes=len(data),
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
    return validation, detected_format


def _pdf_metadata_text(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return None
    return text[:MAX_METADATA_LENGTH]


def _inspect_pdf(data: bytes, settings: Settings) -> PdfInspection:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError(
            f"Could not open PDF ({type(exc).__name__}); the file may be corrupted."
        ) from exc

    try:
        if document.needs_pass or document.is_encrypted:
            raise IngestionError(
                "PDF is password-protected or encrypted and cannot be processed."
            )
        page_count = document.page_count
        if page_count < 1:
            raise IngestionError("PDF contains no pages.")
        if page_count > settings.max_pages_per_pdf:
            raise IngestionError(
                f"PDF has {page_count} pages; the configured limit is "
                f"{settings.max_pages_per_pdf}."
            )

        warnings: list[str] = []
        pages: list[PageInfo] = []
        for index in range(page_count):
            try:
                page = document.load_page(index)
                rect = page.rect
                if rect.width <= 0 or rect.height <= 0:
                    raise IngestionError(
                        f"PDF page {index + 1} has invalid page dimensions."
                    )
                text = page.get_text("text").strip()
                image_count = len(page.get_images(full=True))
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(
                    f"Could not read PDF page {index + 1} ({type(exc).__name__})."
                ) from exc

            if text and image_count:
                content_status = PageContentStatus.MIXED
            elif text:
                content_status = PageContentStatus.TEXT
            elif image_count:
                content_status = PageContentStatus.IMAGE_ONLY
            else:
                content_status = PageContentStatus.EMPTY
                warnings.append(f"Page {index + 1} appears to be empty.")

            pages.append(
                PageInfo(
                    page_number=index + 1,
                    width_pt=float(rect.width),
                    height_pt=float(rect.height),
                    rotation=int(page.rotation) % 360,
                    content_status=content_status,
                    text_characters=len(text),
                    image_count=image_count,
                )
            )

        raw_metadata = document.metadata or {}
        metadata = PdfMetadata(
            title=_pdf_metadata_text(raw_metadata.get("title")),
            author=_pdf_metadata_text(raw_metadata.get("author")),
            subject=_pdf_metadata_text(raw_metadata.get("subject")),
            keywords=_pdf_metadata_text(raw_metadata.get("keywords")),
            creator=_pdf_metadata_text(raw_metadata.get("creator")),
            producer=_pdf_metadata_text(raw_metadata.get("producer")),
            creation_date=_pdf_metadata_text(raw_metadata.get("creationDate")),
            modification_date=_pdf_metadata_text(raw_metadata.get("modDate")),
        )
        return PdfInspection(
            page_count=page_count,
            is_encrypted=bool(document.is_encrypted),
            needs_password=bool(document.needs_pass),
            metadata=metadata,
            pages=tuple(pages),
            warnings=tuple(warnings),
        )
    finally:
        document.close()


def _inspect_image(
    data: bytes,
    expected_format: DocumentFormat,
    settings: Settings,
) -> ImageInfo:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise IngestionError(
            "Image file is corrupted or unsupported and could not be verified."
        ) from exc

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise IngestionError("Image has invalid dimensions.")
            if width * height > settings.max_image_pixels:
                raise IngestionError(
                    f"Image has {width * height} pixels, exceeding the "
                    f"{settings.max_image_pixels} pixel safety limit."
                )
            actual_format = image.format or ""
            if actual_format.upper() == "JPG":
                actual_format = "JPEG"
            if actual_format.upper() != expected_format.value.upper():
                raise IngestionError(
                    f"Image bytes are {actual_format or 'unknown'} but the "
                    f"signature indicated {expected_format.value}."
                )
            exif_orientation: int | None = None
            try:
                exif_orientation = image.getexif().get(274)
            except Exception:
                exif_orientation = None
            if exif_orientation is not None and not 1 <= exif_orientation <= 8:
                exif_orientation = None
            has_transparency = "A" in image.getbands() or "transparency" in image.info
            return ImageInfo(
                width=width,
                height=height,
                mode=image.mode,
                image_format=actual_format.upper(),
                has_transparency=has_transparency,
                exif_orientation=exif_orientation,
            )
    except Image.DecompressionBombError as exc:
        raise IngestionError("Image exceeds Pillow's decompression bomb limit.") from exc
    except OSError as exc:
        raise IngestionError("Image could not be opened after verification.") from exc


def _issue(code: str, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR) -> IngestionIssue:
    return IngestionIssue(code=code, message=message, severity=severity)


def create_document_record(
    result: IngestionResult,
    data: bytes,
    upload_id: str | None = None,
) -> DocumentRecord:
    """Build a safe in-memory record from a validated ingestion result."""
    if result.status is not IngestionStatus.VALIDATED or result.metadata is None:
        raise IngestionError("Cannot create a record from a rejected document.")
    metadata = result.metadata
    return DocumentRecord(
        document_id=result.document_id,
        original_filename=result.original_filename,
        sanitized_filename=metadata.sanitized_filename,
        document_format=metadata.document_format,
        extension=metadata.extension,
        mime_type=metadata.mime_type,
        size_bytes=metadata.size_bytes,
        sha256=metadata.sha256,
        raw_bytes=data,
        metadata=metadata,
        warnings=tuple(issue.message for issue in result.warnings),
        upload_id=upload_id,
    )


def ingest_bytes(
    original_filename: str,
    data: bytes,
    settings: Settings,
    upload_id: str | None = None,
) -> tuple[IngestionResult, DocumentRecord | None]:
    """Validate and inspect one in-memory upload, never raising to the caller."""
    document_id = new_document_id()
    original = original_filename or "unnamed-document"
    try:
        validation, document_format = validate_upload_bytes(original, data, settings)
        if not validation.is_valid:
            issues = tuple(_issue("validation_error", error) for error in validation.errors)
            return (
                IngestionResult(
                    document_id=document_id,
                    original_filename=original,
                    status=IngestionStatus.REJECTED,
                    errors=issues,
                ),
                None,
            )

        if document_format is None:
            raise IngestionError("Could not determine the document format.")

        warnings = tuple(
            _issue("sanitization", warning, ErrorSeverity.WARNING)
            for warning in validation.warnings
        )
        if document_format is DocumentFormat.PDF:
            pdf = _inspect_pdf(data, settings)
            image = None
            warnings += tuple(
                _issue("pdf_page", warning, ErrorSeverity.WARNING)
                for warning in pdf.warnings
            )
        else:
            image = _inspect_image(data, document_format, settings)
            pdf = None

        metadata = DocumentMetadata(
            document_id=document_id,
            original_filename=original,
            sanitized_filename=validation.filename,
            document_format=document_format,
            extension=validation.extension or "",
            mime_type=FORMAT_TO_MIME[document_format],
            size_bytes=len(data),
            sha256=compute_sha256(data),
            image=image,
            pdf=pdf,
        )
        result = IngestionResult(
            document_id=document_id,
            original_filename=original,
            status=IngestionStatus.VALIDATED,
            metadata=metadata,
            warnings=warnings,
        )
        return result, create_document_record(result, data, upload_id)
    except IngestionError as exc:
        logger.warning("Document inspection failed: %s", type(exc).__name__)
        return (
            IngestionResult(
                document_id=document_id,
                original_filename=original,
                status=IngestionStatus.REJECTED,
                errors=(_issue("ingestion_error", str(exc)),),
            ),
            None,
        )
    except Exception as exc:
        logger.warning("Unexpected ingestion failure: %s", type(exc).__name__)
        return (
            IngestionResult(
                document_id=document_id,
                original_filename=original,
                status=IngestionStatus.REJECTED,
                errors=(
                    _issue(
                        "ingestion_error",
                        f"Unexpected {type(exc).__name__} while handling this file.",
                    ),
                ),
            ),
            None,
        )


def ingest_upload(upload: object, settings: Settings) -> tuple[IngestionResult, DocumentRecord | None]:
    """Validate and inspect one upload object from Streamlit or a test double."""
    try:
        data = read_upload_bytes(upload)
    except Exception as exc:
        name = _upload_name(upload)
        message = (
            str(exc)
            if isinstance(exc, IngestionError)
            else f"Uploaded file could not be read ({type(exc).__name__})."
        )
        return (
            IngestionResult(
                document_id=new_document_id(),
                original_filename=name or "unnamed-document",
                status=IngestionStatus.REJECTED,
                errors=(_issue("read_error", message),),
            ),
            None,
        )
    return ingest_bytes(_upload_name(upload), data, settings, _upload_id(upload))


def ingest_uploads(
    uploads: list[object] | tuple[object, ...],
    settings: Settings,
    existing_fingerprints: set[str] | None = None,
) -> BatchIngestionResult:
    """Ingest a batch independently so one bad file never aborts the rest."""
    records: list[DocumentRecord] = []
    rejected: list[IngestionResult] = []
    duplicates: list[str] = []
    skipped_limit: list[str] = []
    seen = set(existing_fingerprints or ())

    for upload in uploads:
        name = _upload_name(upload) or "unnamed-document"
        if len(records) >= settings.max_documents_per_session:
            skipped_limit.append(name)
            continue
        result, record = ingest_upload(upload, settings)
        if record is None:
            rejected.append(result)
            continue
        if record.fingerprint in seen:
            duplicates.append(name)
            continue
        seen.add(record.fingerprint)
        records.append(record)

    return BatchIngestionResult(
        records=tuple(records),
        rejected=tuple(rejected),
        duplicates=tuple(duplicates),
        skipped_limit=tuple(skipped_limit),
    )
