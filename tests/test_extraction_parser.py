"""Tests for parsing and validating untrusted model output."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.extraction.parser import (
    parse_extraction_envelope,
    parse_json_text,
    parse_typed_document,
    validation_error_details,
)
from src.schemas.common import DocumentType
from src.schemas.invoice import InvoiceDocument

from .conftest import make_extraction_settings  # noqa: F401 - fixture availability


def test_parse_json_text_valid() -> None:
    assert parse_json_text('{"a": 1}') == {"a": 1}


def test_parse_json_text_markdown_fence() -> None:
    text = '```json\n{"document_type": "invoice"}\n```'
    assert parse_json_text(text)["document_type"] == "invoice"


def test_parse_json_text_with_surrounding_prose() -> None:
    text = 'Here is the result:\n{"ok": true}\nThanks.'
    assert parse_json_text(text) == {"ok": True}


@pytest.mark.parametrize("text", ["", "   ", "not json at all", '{"a": 1} trailing'])
def test_parse_json_text_rejects_bad_input(text: str) -> None:
    from src.extraction.provider import InvalidResponseError

    with pytest.raises(InvalidResponseError):
        parse_json_text(text)


def test_parse_extraction_envelope_valid() -> None:
    envelope = parse_extraction_envelope(
        """
        {
          "document_type": "invoice",
          "data": {"vendor_name": "Acme"},
          "missing_fields": ["total"],
          "field_confidence": [
            {"field_path": "vendor_name", "confidence": 0.9, "evidence": "Acme"}
          ]
        }
        """
    )
    assert envelope.document_type is DocumentType.INVOICE
    assert envelope.data["vendor_name"] == "Acme"
    assert envelope.field_confidence[0].evidence == "Acme"


def test_parse_extraction_envelope_rejects_extra_fields() -> None:
    from src.extraction.provider import InvalidResponseError

    with pytest.raises(InvalidResponseError):
        parse_extraction_envelope('{"document_type":"invoice","surprise":1}')


def test_parse_typed_document_invoice() -> None:
    payload = parse_typed_document(
        {
            "vendor_name": "Acme",
            "currency": "USD",
            "total": "12.50",
            "invoice_date": "2026-01-02",
        },
        DocumentType.INVOICE,
        document_id="doc-1",
        source_filename="invoice.png",
        extracted_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        warnings=["check total"],
        missing_fields=["invoice_number"],
        unreadable_fields=["tax"],
    )
    assert isinstance(payload, InvoiceDocument)
    assert payload.document_id == "doc-1"
    assert payload.source_filename == "invoice.png"
    assert payload.total == 12.5
    assert payload.missing_fields == ["invoice_number"]
    assert payload.warnings == ["check total"]


def test_parse_typed_document_rejects_invalid_decimal() -> None:
    with pytest.raises(ValidationError):
        parse_typed_document(
            {"total": 12.5},
            DocumentType.INVOICE,
            document_id="doc-1",
            source_filename="invoice.png",
            extracted_at=datetime.now(timezone.utc),
            warnings=[],
            missing_fields=[],
            unreadable_fields=[],
        )


def test_parse_typed_document_unknown_returns_none() -> None:
    assert (
        parse_typed_document(
            {},
            DocumentType.UNKNOWN,
            document_id="doc-1",
            source_filename="x.png",
            extracted_at=datetime.now(timezone.utc),
            warnings=[],
            missing_fields=[],
            unreadable_fields=[],
        )
        is None
    )


def test_validation_error_details_hide_raw_input() -> None:
    try:
        InvoiceDocument.model_validate(
            {
                "document_id": "doc-1",
                "source_filename": "secret-name.png",
                "extracted_at": datetime.now(timezone.utc),
                "total": 12.5,
            }
        )
    except ValidationError as exc:
        details = validation_error_details(exc)
    assert details
    assert all("secret-name.png" not in detail.message for detail in details)
    assert all(detail.field_path for detail in details)
