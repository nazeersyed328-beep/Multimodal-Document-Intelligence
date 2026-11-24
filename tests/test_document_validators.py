"""Unit tests for document preflight validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document.validators import (
    DocumentValidationError,
    detect_mime_type,
    extract_extension,
    validate_document,
    validate_extension,
    validate_filename,
    validate_file_size,
    validate_not_empty,
    validate_signature,
)

PDF_HEADER = b"%PDF-1.7\n" + b"\x00" * 32
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 32


def write_document(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_extract_extension() -> None:
    assert extract_extension("invoice.PDF") == ".pdf"
    assert extract_extension("no_extension") == ""


def test_supported_extension_accepted() -> None:
    assert validate_extension("photo.webp") == ".webp"


def test_unsupported_extension_rejected() -> None:
    with pytest.raises(DocumentValidationError, match="Unsupported file extension"):
        validate_extension("malware.exe")


def test_missing_extension_rejected() -> None:
    with pytest.raises(DocumentValidationError, match="no extension"):
        validate_extension("document")


def test_filename_safety() -> None:
    validate_filename("invoice-01.pdf")
    with pytest.raises(DocumentValidationError):
        validate_filename("")
    with pytest.raises(DocumentValidationError):
        validate_filename("../invoice.pdf")
    with pytest.raises(DocumentValidationError):
        validate_filename("folder\\invoice.pdf")
    with pytest.raises(DocumentValidationError):
        validate_filename("CON.pdf")


def test_empty_file_rejected() -> None:
    with pytest.raises(DocumentValidationError, match="empty"):
        validate_not_empty(0)


def test_file_size_limits() -> None:
    validate_file_size(100, 200)
    with pytest.raises(DocumentValidationError, match="exceeds"):
        validate_file_size(201, 200)


@pytest.mark.parametrize(
    ("header", "extension"),
    [
        (PDF_HEADER, ".pdf"),
        (PNG_HEADER, ".png"),
        (JPEG_HEADER, ".jpg"),
        (JPEG_HEADER, ".jpeg"),
        (WEBP_HEADER, ".webp"),
    ],
)
def test_valid_signatures(header: bytes, extension: str) -> None:
    validate_signature(header, extension)


def test_signature_mismatch_rejected() -> None:
    with pytest.raises(DocumentValidationError, match="signature does not match"):
        validate_signature(PNG_HEADER, ".pdf")


def test_mime_detection() -> None:
    assert detect_mime_type(PNG_HEADER) == "image/png"
    assert detect_mime_type(JPEG_HEADER) == "image/jpeg"
    assert detect_mime_type(PDF_HEADER) == "application/pdf"
    assert detect_mime_type(WEBP_HEADER) == "image/webp"
    assert detect_mime_type(b"random bytes") is None


def test_validate_document_accepts_valid_png(tmp_path: Path) -> None:
    path = write_document(tmp_path, "page.png", PNG_HEADER)
    result = validate_document(path, max_size_bytes=1024 * 1024)
    assert result.is_valid is True
    assert result.extension == ".png"
    assert result.size_bytes == len(PNG_HEADER)
    assert result.errors == ()


def test_validate_document_accepts_valid_pdf(tmp_path: Path) -> None:
    path = write_document(tmp_path, "invoice.pdf", PDF_HEADER)
    result = validate_document(path, max_size_bytes=1024 * 1024)
    assert result.is_valid is True
    assert result.extension == ".pdf"


def test_validate_document_rejects_empty_file(tmp_path: Path) -> None:
    path = write_document(tmp_path, "empty.png", b"")
    result = validate_document(path, max_size_bytes=1024 * 1024)
    assert result.is_valid is False
    assert any("empty" in error.lower() for error in result.errors)


def test_validate_document_rejects_unsupported_format(tmp_path: Path) -> None:
    path = write_document(tmp_path, "notes.txt", b"plain text")
    result = validate_document(path, max_size_bytes=1024 * 1024)
    assert result.is_valid is False
    assert any("Unsupported file extension" in error for error in result.errors)


def test_validate_document_rejects_signature_mismatch(tmp_path: Path) -> None:
    path = write_document(tmp_path, "fake.pdf", PNG_HEADER)
    result = validate_document(path, max_size_bytes=1024 * 1024)
    assert result.is_valid is False
    assert any("signature" in error.lower() for error in result.errors)


def test_validate_document_rejects_oversized_file(tmp_path: Path) -> None:
    path = write_document(tmp_path, "large.png", PNG_HEADER)
    result = validate_document(path, max_size_bytes=10)
    assert result.is_valid is False
    assert any("exceeds" in error for error in result.errors)


def test_validate_document_rejects_missing_file(tmp_path: Path) -> None:
    result = validate_document(tmp_path / "missing.png", max_size_bytes=1024)
    assert result.is_valid is False
    assert "File does not exist." in result.errors
