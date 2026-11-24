"""Preflight validation for candidate document files.

Only structural checks are performed here: filename safety, extension, size,
emptiness, MIME consistency, and basic file signatures. No OCR, rendering, or
AI extraction happens in this module.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_ALLOWED_EXTENSIONS

SIGNATURE_LENGTH = 16
RESERVED_WINDOWS_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
PDF_SIGNATURE = b"%PDF-"


class DocumentValidationError(ValueError):
    """Raised by an individual preflight check when a file is not acceptable."""


@dataclass(frozen=True)
class DocumentValidationResult:
    """Aggregated outcome of all preflight checks for one file."""

    filename: str
    extension: str | None
    size_bytes: int | None
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def extract_extension(filename: str) -> str:
    """Return the lowercase dotted suffix of a filename, or an empty string."""
    return Path(filename).suffix.lower()


def normalize_extensions(extensions: Iterable[str]) -> tuple[str, ...]:
    """Normalize and deduplicate an iterable of dotted file extensions."""
    normalized: list[str] = []
    for extension in extensions:
        candidate = extension.strip().lower()
        if not candidate:
            continue
        if not candidate.startswith("."):
            candidate = f".{candidate}"
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def validate_filename(filename: str) -> None:
    """Reject empty, oversized, path-like, or Windows-reserved filenames."""
    if not filename or not filename.strip():
        raise DocumentValidationError("Filename must not be empty.")
    if len(filename) > 255:
        raise DocumentValidationError("Filename must not exceed 255 characters.")
    if filename in {".", ".."}:
        raise DocumentValidationError(f"Filename {filename!r} is not a valid file name.")
    if any(char in filename for char in ("/", "\\", "\x00")):
        raise DocumentValidationError(
            "Filename must not contain path separators or null characters."
        )
    if any(ord(char) < 32 for char in filename):
        raise DocumentValidationError("Filename must not contain control characters.")
    if Path(filename).stem.upper() in RESERVED_WINDOWS_NAMES:
        raise DocumentValidationError(
            f"Filename {filename!r} uses a reserved Windows device name."
        )


def validate_extension(
    filename: str,
    allowed_extensions: Iterable[str] = DEFAULT_ALLOWED_EXTENSIONS,
) -> str:
    """Validate the file extension and return its normalized form."""
    extension = extract_extension(filename)
    allowed = normalize_extensions(allowed_extensions)
    if not extension:
        raise DocumentValidationError(
            f"File {filename!r} has no extension; expected one of: {', '.join(allowed)}."
        )
    if extension not in allowed:
        raise DocumentValidationError(
            f"Unsupported file extension {extension!r}; "
            f"allowed extensions are: {', '.join(allowed)}."
        )
    return extension


def validate_not_empty(size_bytes: int) -> None:
    """Reject a zero-byte file."""
    if size_bytes <= 0:
        raise DocumentValidationError("File is empty (0 bytes).")


def validate_file_size(size_bytes: int, max_size_bytes: int) -> None:
    """Reject negative sizes and files larger than the configured limit."""
    if size_bytes < 0:
        raise DocumentValidationError("File size cannot be negative.")
    if max_size_bytes <= 0:
        raise DocumentValidationError("Maximum upload size must be greater than zero.")
    if size_bytes > max_size_bytes:
        raise DocumentValidationError(
            f"File size {size_bytes} bytes exceeds the "
            f"{max_size_bytes} byte upload limit."
        )


def detect_mime_type(header: bytes) -> str | None:
    """Identify a supported MIME type from a leading byte header."""
    if header.startswith(PDF_SIGNATURE):
        return "application/pdf"
    if header.startswith(PNG_SIGNATURE):
        return "image/png"
    if header.startswith(JPEG_SIGNATURE):
        return "image/jpeg"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_signature(header: bytes, extension: str) -> None:
    """Check that leading bytes match the signature expected for the type."""
    extension = extension.lower()
    if extension == ".pdf":
        expected = "PDF"
        matches = header.startswith(PDF_SIGNATURE)
    elif extension == ".png":
        expected = "PNG"
        matches = header.startswith(PNG_SIGNATURE)
    elif extension in {".jpg", ".jpeg"}:
        expected = "JPEG"
        matches = header.startswith(JPEG_SIGNATURE)
    elif extension == ".webp":
        expected = "WEBP"
        matches = header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    else:
        raise DocumentValidationError(
            f"No file signature check is defined for extension {extension!r}."
        )
    if not matches:
        raise DocumentValidationError(
            f"File signature does not match the expected {expected} format."
        )


def validate_mime_type(filename: str, header: bytes) -> str | None:
    """Compare the MIME guessed from the name with the MIME sniffed from bytes."""
    detected = detect_mime_type(header)
    guessed = mimetypes.guess_type(filename)[0]
    if detected is None:
        raise DocumentValidationError(
            "File signature does not match any supported MIME type."
        )
    if guessed is not None and guessed != detected:
        raise DocumentValidationError(
            f"MIME mismatch: filename suggests {guessed!r} but bytes indicate "
            f"{detected!r}."
        )
    return detected


def validate_document(
    path: str | Path,
    *,
    max_size_bytes: int,
    allowed_extensions: Iterable[str] = DEFAULT_ALLOWED_EXTENSIONS,
) -> DocumentValidationResult:
    """Run every preflight check and collect all problems into one result."""
    file_path = Path(path)
    filename = file_path.name
    errors: list[str] = []
    warnings: list[str] = []
    extension: str | None = None
    size_bytes: int | None = None

    try:
        validate_filename(filename)
    except DocumentValidationError as exc:
        errors.append(str(exc))

    try:
        extension = validate_extension(filename, allowed_extensions)
    except DocumentValidationError as exc:
        errors.append(str(exc))

    if not file_path.exists():
        errors.append("File does not exist.")
    elif not file_path.is_file():
        errors.append("Path is not a regular file.")
    else:
        try:
            size_bytes = file_path.stat().st_size
        except OSError as exc:
            errors.append(f"Could not read file metadata: {exc}")
        else:
            size_ok = True
            try:
                validate_not_empty(size_bytes)
            except DocumentValidationError as exc:
                errors.append(str(exc))
                size_ok = False
            try:
                validate_file_size(size_bytes, max_size_bytes)
            except DocumentValidationError as exc:
                errors.append(str(exc))
                size_ok = False

            if size_ok and extension is not None:
                try:
                    with file_path.open("rb") as handle:
                        header = handle.read(SIGNATURE_LENGTH)
                except OSError as exc:
                    errors.append(f"Could not read file header: {exc}")
                else:
                    if not header:
                        errors.append("File is empty (0 bytes).")
                    else:
                        try:
                            validate_signature(header, extension)
                        except DocumentValidationError as exc:
                            errors.append(str(exc))
                        else:
                            try:
                                validate_mime_type(filename, header)
                            except DocumentValidationError as exc:
                                errors.append(str(exc))
                    if len(header) < SIGNATURE_LENGTH:
                        warnings.append(
                            "File is shorter than the signature inspection window."
                        )

    return DocumentValidationResult(
        filename=filename,
        extension=extension,
        size_bytes=size_bytes,
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
