"""Document upload, validation, inspection, preprocessing, and session helpers."""

from .ingestion import (
    BatchIngestionResult,
    DocumentRecord,
    IngestionError,
    compute_sha256,
    create_document_record,
    detect_format,
    ingest_bytes,
    ingest_upload,
    ingest_uploads,
    new_document_id,
    read_upload_bytes,
    sanitize_filename,
    validate_upload_bytes,
)
from .preprocessing import (
    NormalizedImage,
    PreprocessingError,
    clear_processed_bytes,
    make_preview,
    make_preview_from_bytes,
    normalize_image,
    preprocess_document,
    render_pdf_page,
)
from .session import DocumentSession

__all__ = [
    "BatchIngestionResult",
    "DocumentRecord",
    "DocumentSession",
    "IngestionError",
    "NormalizedImage",
    "PreprocessingError",
    "compute_sha256",
    "clear_processed_bytes",
    "create_document_record",
    "detect_format",
    "ingest_bytes",
    "ingest_upload",
    "ingest_uploads",
    "make_preview",
    "make_preview_from_bytes",
    "new_document_id",
    "normalize_image",
    "preprocess_document",
    "read_upload_bytes",
    "render_pdf_page",
    "sanitize_filename",
    "validate_upload_bytes",
]
