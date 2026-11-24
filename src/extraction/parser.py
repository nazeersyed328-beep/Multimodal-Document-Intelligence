"""Safe parsing of untrusted vision-model output."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from src.schemas.common import DocumentType, ErrorDetail, ValidationStatus
from src.schemas.diagram import DiagramDocument
from src.schemas.extraction import ExtractionEnvelope
from src.schemas.extraction_result import DocumentPayload
from src.schemas.handwritten_note import HandwrittenNoteDocument
from src.schemas.invoice import InvoiceDocument
from src.schemas.receipt import ReceiptDocument

from .provider import InvalidResponseError

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_DOCUMENT_MODELS = {
    DocumentType.INVOICE: InvoiceDocument,
    DocumentType.RECEIPT: ReceiptDocument,
    DocumentType.HANDWRITTEN_NOTE: HandwrittenNoteDocument,
    DocumentType.DIAGRAM: DiagramDocument,
}


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text.strip()


def parse_json_text(text: str | None) -> Any:
    """Parse a JSON object/array, allowing Markdown fences and surrounding prose."""
    if not text or not text.strip():
        raise InvalidResponseError("The model returned no text.")

    candidate = _strip_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    starts = [index for marker in ("{", "[") if (index := candidate.find(marker)) >= 0]
    if not starts:
        raise InvalidResponseError("The model response did not contain JSON.")

    decoder = json.JSONDecoder()
    for start in sorted(starts):
        try:
            value, end = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        remainder = candidate[start + end :]
        if remainder.strip() and "\n" not in remainder:
            continue
        return value
    raise InvalidResponseError("The model response contained malformed JSON.")


def _error_location(error: dict[str, Any]) -> str:
    return ".".join(str(part) for part in error.get("loc", ())) or "root"


def validation_error_details(exc: ValidationError) -> list[ErrorDetail]:
    """Convert Pydantic errors into safe ErrorDetail values without raw inputs."""
    details: list[ErrorDetail] = []
    for error in exc.errors():
        message = str(error.get("msg", "Invalid value."))
        details.append(
            ErrorDetail(
                code=str(error.get("type", "validation_error"))[:100],
                message=message[:2000],
                field_path=_error_location(error)[:500],
            )
        )
    return details


def validation_error_messages(exc: ValidationError) -> list[str]:
    """Return short, actionable messages for a bounded correction prompt."""
    return [
        f"{detail.field_path}: {detail.message}"
        for detail in validation_error_details(exc)[:20]
    ]


def parse_extraction_envelope(text: str | None) -> ExtractionEnvelope:
    """Parse and validate a model response into the extraction envelope."""
    payload = parse_json_text(text)
    try:
        return ExtractionEnvelope.model_validate(payload)
    except ValidationError as exc:
        messages = validation_error_messages(exc)
        raise InvalidResponseError(
            "Model output failed envelope validation: " + "; ".join(messages[:5])
        ) from None


def parse_typed_document(
    data: dict[str, Any],
    document_type: DocumentType,
    *,
    document_id: str,
    source_filename: str,
    extracted_at: datetime,
    warnings: list[str],
    missing_fields: list[str],
    unreadable_fields: list[str],
) -> DocumentPayload | None:
    """Validate envelope data against the strict schema for its document type."""
    model = _DOCUMENT_MODELS.get(document_type)
    if model is None:
        return None

    controlled = dict(data)
    controlled["document_id"] = document_id
    controlled["document_type"] = document_type.value
    controlled["source_filename"] = source_filename
    controlled["extracted_at"] = extracted_at
    controlled["validation_status"] = ValidationStatus.NOT_VALIDATED
    controlled["warnings"] = warnings
    controlled["errors"] = []
    controlled["missing_fields"] = missing_fields
    controlled["unreadable_fields"] = unreadable_fields
    return model.model_validate(controlled)


__all__ = [
    "parse_extraction_envelope",
    "parse_json_text",
    "parse_typed_document",
    "validation_error_details",
    "validation_error_messages",
]
