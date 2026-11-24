"""Tests for deterministic consistency rules."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from src.config import Settings
from src.quality import (
    collect_diagram_issues,
    collect_invoice_issues,
    collect_note_issues,
    collect_receipt_issues,
)
from src.schemas.common import FieldConfidenceEntry
from src.schemas.diagram import (
    Ambiguity,
    DiagramConnection,
    DiagramDocument,
    DiagramElement,
    DiagramType,
)
from src.schemas.handwritten_note import (
    ActionItem,
    DateMention,
    HandwrittenNoteDocument,
    UnreadableMarker,
)
from src.schemas.invoice import InvoiceDocument, InvoiceLineItem
from src.schemas.receipt import ReceiptDocument, ReceiptItem

from .conftest import make_extraction_settings

NOW = datetime(2026, 1, 3, tzinfo=timezone.utc)
TODAY = date(2026, 1, 15)


def settings(**overrides: str) -> Settings:
    return make_extraction_settings(**overrides)


def invoice(**overrides: object) -> InvoiceDocument:
    values: dict[str, object] = {
        "document_id": "doc-invoice",
        "source_filename": "invoice.png",
        "extracted_at": NOW,
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": date(2026, 1, 2),
        "due_date": date(2026, 1, 20),
        "currency": "USD",
        "subtotal": Decimal("10.00"),
        "tax": Decimal("1.00"),
        "total": Decimal("11.00"),
        "line_items": [
            InvoiceLineItem(
                description="Widget",
                quantity=Decimal("2"),
                unit_price=Decimal("5.00"),
                amount=Decimal("10.00"),
            )
        ],
    }
    values.update(overrides)
    return InvoiceDocument(**values)  # type: ignore[arg-type]


def receipt(**overrides: object) -> ReceiptDocument:
    values: dict[str, object] = {
        "document_id": "doc-receipt",
        "source_filename": "receipt.jpg",
        "extracted_at": NOW,
        "merchant_name": "Corner Shop",
        "transaction_datetime": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "currency": "USD",
        "items": [
            ReceiptItem(
                description="Milk",
                quantity=Decimal("2"),
                unit_price=Decimal("3.00"),
                item_amount=Decimal("6.00"),
            )
        ],
        "subtotal": Decimal("6.00"),
        "tax": Decimal("0.50"),
        "total": Decimal("6.50"),
    }
    values.update(overrides)
    return ReceiptDocument(**values)  # type: ignore[arg-type]


def note(**overrides: object) -> HandwrittenNoteDocument:
    values: dict[str, object] = {
        "document_id": "doc-note",
        "source_filename": "note.jpg",
        "extracted_at": NOW,
        "transcribed_text": "Call Sam tomorrow.",
        "language": "en",
    }
    values.update(overrides)
    return HandwrittenNoteDocument(**values)  # type: ignore[arg-type]


def diagram(**overrides: object) -> DiagramDocument:
    values: dict[str, object] = {
        "document_id": "doc-diagram",
        "source_filename": "diagram.png",
        "extracted_at": NOW,
        "title": "Flow",
        "diagram_type": DiagramType.FLOWCHART,
        "elements": [
            DiagramElement(element_id="a", label="Start"),
            DiagramElement(element_id="b", label="End"),
        ],
        "connections": [
            DiagramConnection(source_id="a", target_id="b", label="then")
        ],
    }
    values.update(overrides)
    return DiagramDocument(**values)  # type: ignore[arg-type]


def test_clean_invoice_has_no_consistency_issues() -> None:
    assert collect_invoice_issues(invoice(), settings(), TODAY) == []


def test_invoice_line_math_is_checked() -> None:
    data = invoice(
        line_items=[
            InvoiceLineItem(
                description="Widget",
                quantity=Decimal("2"),
                unit_price=Decimal("5.00"),
                amount=Decimal("12.00"),
            )
        ],
        subtotal=Decimal("12.00"),
        total=Decimal("13.00"),
    )
    rules = {issue.rule_id for issue in collect_invoice_issues(data, settings(), TODAY)}
    assert "INV-LINE-001" in rules


def test_invoice_subtotal_and_total_math_are_checked() -> None:
    data = invoice(subtotal=Decimal("9.00"))
    rules = {issue.rule_id for issue in collect_invoice_issues(data, settings(), TODAY)}
    assert "INV-TOTAL-001" in rules
    assert "INV-TOTAL-002" in rules


def test_invoice_discount_is_used_without_an_assumed_tax_rate() -> None:
    data = invoice(discount=Decimal("1.00"), total=Decimal("10.00"))
    assert collect_invoice_issues(data, settings(), TODAY) == []


def test_small_amount_difference_is_within_tolerance() -> None:
    data = invoice(total=Decimal("11.005"))
    assert collect_invoice_issues(data, settings(), TODAY) == []


def test_far_future_invoice_date_is_flagged() -> None:
    data = invoice(invoice_date=date(2030, 1, 1), due_date=date(2030, 1, 10))
    rules = {issue.rule_id for issue in collect_invoice_issues(data, settings(), TODAY)}
    assert "DOC-DATE-002" in rules


def test_clean_receipt_has_no_consistency_issues() -> None:
    assert collect_receipt_issues(receipt(), settings(), TODAY) == []


def test_receipt_total_and_subtotal_are_checked() -> None:
    data = receipt(subtotal=Decimal("5.00"), total=Decimal("9.00"))
    rules = {issue.rule_id for issue in collect_receipt_issues(data, settings(), TODAY)}
    assert "RCPT-TOTAL-001" in rules
    assert "RCPT-TOTAL-004" in rules


def test_receipt_missing_total_and_date_are_flagged() -> None:
    data = receipt(total=None, transaction_datetime=None)
    rules = {issue.rule_id for issue in collect_receipt_issues(data, settings(), TODAY)}
    assert "RCPT-TOTAL-002" in rules
    assert "RCPT-DATE-001" in rules


def test_note_unreadable_marker_is_flagged() -> None:
    data = note(unreadable_markers=[UnreadableMarker(marker_id="m1", context="name")])
    rules = {issue.rule_id for issue in collect_note_issues(data, {}, 0.7)}
    assert "NOTE-002" in rules


def test_note_low_transcription_confidence_is_flagged() -> None:
    data = note()
    confidence = {"transcribed_text": FieldConfidenceEntry(value=data.transcribed_text, confidence=0.4)}
    rules = {issue.rule_id for issue in collect_note_issues(data, confidence, 0.7)}
    assert "NOTE-004" in rules


def test_note_interpretation_without_transcription_is_escalated() -> None:
    data = note(
        transcribed_text=None,
        language=None,
        key_points=["Buy milk"],
        action_items=[ActionItem(text="Call Sam")],
        dates_mentioned=[DateMention(raw_text="tomorrow")],
    )
    rules = {issue.rule_id for issue in collect_note_issues(data, {}, 0.7)}
    assert "NOTE-008" in rules
    assert "NOTE-006" in rules


def test_diagram_ambiguity_is_flagged() -> None:
    data = diagram(ambiguities=[Ambiguity(description="Arrow direction is unclear.")])
    rules = {issue.rule_id for issue in collect_diagram_issues(data)}
    assert "DIAG-010" in rules


def test_diagram_unlabeled_connection_is_flagged() -> None:
    data = diagram(
        connections=[DiagramConnection(source_id="a", target_id="b", label="unclear")]
    )
    rules = {issue.rule_id for issue in collect_diagram_issues(data)}
    assert "DIAG-006" in rules


def test_diagram_missing_title_and_elements_are_flagged() -> None:
    data = diagram(title=None, elements=[], connections=[])
    rules = {issue.rule_id for issue in collect_diagram_issues(data)}
    assert "DIAG-001" in rules
    assert "DIAG-002" in rules
