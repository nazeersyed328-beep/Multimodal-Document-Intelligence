"""Tests for deterministic, safe extraction result export."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

from src.extraction.export import SCHEMA_VERSION, dump_extraction_result, extraction_result_to_dict
from src.schemas.common import DocumentType, FieldConfidenceEntry, ValidationStatus
from src.schemas.extraction_result import ExtractionResult
from src.schemas.invoice import InvoiceDocument


def make_result() -> ExtractionResult:
    payload = InvoiceDocument(
        document_id="doc-1",
        document_type=DocumentType.INVOICE,
        source_filename="invoice.png",
        extracted_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        validation_status=ValidationStatus.NOT_VALIDATED,
        vendor_name="Acme",
        invoice_date=date(2026, 1, 2),
        currency="USD",
        total=Decimal("12.50"),
    )
    return ExtractionResult(
        result_id="result-1",
        document_type=DocumentType.INVOICE,
        source_document_id="doc-1",
        data=payload,
        field_confidence={
            "total": FieldConfidenceEntry(value=Decimal("12.50"), confidence=0.91)
        },
        overall_confidence=0.91,
    )


def test_export_is_json_serializable() -> None:
    text = dump_extraction_result(make_result())
    parsed = json.loads(text)

    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["extraction"]["result_id"] == "result-1"
    assert parsed["extraction"]["data"]["total"] == "12.50"
    assert parsed["extraction"]["data"]["invoice_date"] == "2026-01-02"
    assert parsed["extraction"]["field_confidence"]["total"]["value"] == "12.50"


def test_export_contains_no_raw_document_bytes() -> None:
    serialized = dump_extraction_result(make_result())
    assert "image_bytes" not in serialized
    assert "raw_bytes" not in serialized


def test_export_dict_is_independent_copy() -> None:
    result = make_result()
    exported = extraction_result_to_dict(result)
    exported["extraction"]["result_id"] = "changed"
    assert result.result_id == "result-1"
