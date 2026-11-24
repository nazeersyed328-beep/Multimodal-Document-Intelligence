"""Unit tests for the strict Pydantic schemas."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.schemas.common import (
    DocumentType,
    FieldConfidence,
    FieldConfidenceEntry,
    ValidationStatus,
)
from src.schemas.diagram import DiagramConnection, DiagramDocument, DiagramElement
from src.schemas.extraction_result import ExtractionResult, ProcessingStatus
from src.schemas.handwritten_note import HandwrittenNoteDocument, UnreadableMarker
from src.schemas.invoice import InvoiceDocument, InvoiceLineItem
from src.schemas.receipt import ReceiptDocument, ReceiptItem

BASE = {
    "document_id": "doc-001",
    "source_filename": "sample.pdf",
    "extracted_at": datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
}


def make_invoice(**overrides: object) -> InvoiceDocument:
    return InvoiceDocument(**{**BASE, **overrides})


def test_document_type_enum_values() -> None:
    assert {item.value for item in DocumentType} == {
        "invoice",
        "receipt",
        "handwritten_note",
        "diagram",
        "unknown",
    }


def test_invoice_creation_with_line_item() -> None:
    invoice = make_invoice(
        vendor_name="Acme Ltd",
        currency="USD",
        invoice_date=date(2026, 1, 1),
        due_date=date(2026, 1, 31),
        total=Decimal("120.50"),
        line_items=[
            InvoiceLineItem(
                description="Widget",
                quantity=Decimal("2"),
                unit_price=Decimal("50.00"),
                amount=Decimal("100.00"),
            )
        ],
    )
    assert invoice.document_type is DocumentType.INVOICE
    assert invoice.total == Decimal("120.50")
    assert invoice.line_items[0].quantity == Decimal("2")
    assert invoice.validation_status is ValidationStatus.NOT_VALIDATED


def test_optional_fields_default_to_none() -> None:
    invoice = make_invoice()
    assert invoice.vendor_name is None
    assert invoice.currency is None
    assert invoice.invoice_date is None
    assert invoice.line_items == []
    assert invoice.missing_fields == []
    assert invoice.unreadable_fields == []


def test_negative_monetary_value_rejected() -> None:
    with pytest.raises(ValidationError):
        make_invoice(total=Decimal("-1.00"))


def test_floating_point_monetary_value_rejected() -> None:
    with pytest.raises(ValidationError, match="Floating-point"):
        make_invoice(total=12.5)


def test_numeric_string_with_thousands_separator_accepted() -> None:
    item = InvoiceLineItem(description="Bulk order", amount="1,234.56")
    assert item.amount == Decimal("1234.56")


def test_invalid_currency_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_invoice(currency="usd")
    assert "currency" in str(exc_info.value)


def test_invalid_date_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_invoice(invoice_date="2026-02-30")
    assert "invoice_date" in str(exc_info.value)


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        InvoiceLineItem(description="Widget", quantity=Decimal("0"))


def test_due_date_before_invoice_date_rejected() -> None:
    with pytest.raises(ValidationError, match="due_date cannot be earlier"):
        make_invoice(
            invoice_date=date(2026, 3, 10),
            due_date=date(2026, 3, 1),
        )


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_invoice(surprise="not in schema")


def test_source_filename_rejects_path() -> None:
    with pytest.raises(ValidationError):
        make_invoice(source_filename="../secret.pdf")


def test_receipt_creation() -> None:
    receipt = ReceiptDocument(
        **BASE,
        merchant_name="Corner Shop",
        currency="EUR",
        transaction_datetime="2026-01-15T09:30:00",
        items=[
            ReceiptItem(
                description="Coffee",
                quantity=Decimal("2"),
                unit_price=Decimal("3.50"),
                item_amount=Decimal("7.00"),
            )
        ],
        tax=Decimal("0.70"),
        total=Decimal("7.70"),
        payment_method="CASH",
    )
    assert receipt.document_type is DocumentType.RECEIPT
    assert receipt.transaction_datetime is not None
    assert receipt.transaction_datetime.year == 2026
    assert receipt.items[0].item_amount == Decimal("7.00")


def test_receipt_negative_tax_rejected() -> None:
    with pytest.raises(ValidationError):
        ReceiptDocument(**BASE, tax=Decimal("-0.50"))


def test_handwritten_note_unreadable_markers() -> None:
    note = HandwrittenNoteDocument(
        **BASE,
        title="Meeting",
        transcribed_text="Call [UNREADABLE_1] tomorrow",
        language="en",
        key_points=["Follow up"],
        unreadable_markers=[
            UnreadableMarker(marker_id="UNREADABLE_1", reason="illegible")
        ],
    )
    assert note.unreadable_markers[0].marker_id == "UNREADABLE_1"
    assert note.transcribed_text == "Call [UNREADABLE_1] tomorrow"


def test_duplicate_unreadable_marker_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        HandwrittenNoteDocument(
            **BASE,
            unreadable_markers=[
                UnreadableMarker(marker_id="M1"),
                UnreadableMarker(marker_id="M1"),
            ],
        )


def test_invalid_language_code_rejected() -> None:
    with pytest.raises(ValidationError):
        HandwrittenNoteDocument(**BASE, language="English")


def test_diagram_creation_and_connections() -> None:
    diagram = DiagramDocument(
        **BASE,
        title="Order flow",
        elements=[
            DiagramElement(element_id="start", label="Start"),
            DiagramElement(element_id="end", label="End"),
        ],
        connections=[
            DiagramConnection(
                source_id="start",
                target_id="end",
                relationship_type="next",
            )
        ],
        labels=["Start", "End"],
        extracted_text=["Start", "End"],
    )
    assert diagram.document_type is DocumentType.DIAGRAM
    assert len(diagram.elements) == 2
    assert diagram.connections[0].target_id == "end"


def test_diagram_unknown_connection_target_rejected() -> None:
    with pytest.raises(ValidationError, match="does not reference a known"):
        DiagramDocument(
            **BASE,
            elements=[DiagramElement(element_id="start")],
            connections=[
                DiagramConnection(source_id="start", target_id="missing")
            ],
        )


def test_confidence_bounds() -> None:
    assert FieldConfidence[str](value="x", confidence=0.0).confidence == 0.0
    assert FieldConfidence[str](value="x", confidence=1.0).confidence == 1.0
    with pytest.raises(ValidationError):
        FieldConfidence[str](value="x", confidence=-0.01)
    with pytest.raises(ValidationError):
        FieldConfidence[str](value="x", confidence=1.01)


def test_field_confidence_rejects_arbitrary_object() -> None:
    with pytest.raises(ValidationError):
        FieldConfidenceEntry(value={"unstructured": "object"})


def test_extraction_result_with_typed_payload() -> None:
    invoice = make_invoice(vendor_name="Acme Ltd")
    result = ExtractionResult(
        result_id="result-001",
        document_type=DocumentType.INVOICE,
        status=ProcessingStatus.COMPLETED,
        data=invoice,
        field_confidence={
            "vendor_name": FieldConfidence[str](
                value="Acme Ltd",
                confidence=0.8,
                evidence="Acme Ltd",
            )
        },
    )
    assert result.data is not None
    assert result.data.document_type is DocumentType.INVOICE
    assert result.overall_confidence is None
    assert result.needs_review is False
    assert result.field_confidence["vendor_name"].confidence == 0.8


def test_extraction_result_rejects_mismatched_payload_type() -> None:
    with pytest.raises(ValidationError, match="must match document_type"):
        ExtractionResult(
            result_id="result-002",
            document_type=DocumentType.RECEIPT,
            data=make_invoice(),
        )


def test_extraction_result_rejects_arbitrary_data() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(
            result_id="result-003",
            document_type=DocumentType.INVOICE,
            data={"arbitrary": "object"},
        )


def test_extraction_result_confidence_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(
            result_id="result-004",
            document_type=DocumentType.UNKNOWN,
            overall_confidence=2.0,
        )
