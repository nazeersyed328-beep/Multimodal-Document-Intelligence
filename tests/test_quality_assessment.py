"""Tests for structured quality reports and status rules."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.document.ingestion import ingest_upload
from src.document.preprocessing import preprocess_document
from src.quality import QualityAssessmentService, calculate_quality_score
from src.schemas.common import (
    DocumentType,
    ErrorDetail,
    FieldConfidenceEntry,
    ValidationStatus,
)
from src.schemas.extraction_result import ProcessingStatus
from src.schemas.invoice import InvoiceDocument, InvoiceLineItem
from src.schemas.quality import (
    QualityCategory,
    QualityIssue,
    QualitySeverity,
    QualityStatus,
)

from .conftest import FakeUpload, make_extraction_settings, make_pdf, make_result

TODAY = date(2026, 1, 15)
NOW = datetime(2026, 1, 3, tzinfo=timezone.utc)


def invoice(**overrides: object) -> InvoiceDocument:
    values: dict[str, object] = {
        "document_id": "doc-1",
        "source_filename": "invoice.png",
        "extracted_at": NOW,
        "vendor_name": "Acme Supplies",
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


def confident_result(data: object) -> object:
    return make_result(
        data,
        field_confidence={
            "vendor_name": FieldConfidenceEntry(value="Acme Supplies", confidence=0.95, evidence="Acme Supplies"),
            "invoice_number": FieldConfidenceEntry(value="INV-1", confidence=0.94),
            "invoice_date": FieldConfidenceEntry(value=date(2026, 1, 2), confidence=0.96),
            "currency": FieldConfidenceEntry(value="USD", confidence=0.99),
            "total": FieldConfidenceEntry(value=Decimal("11.00"), confidence=0.97),
        },
    )


def assess(data: object, result: object | None = None, **settings: str) -> object:
    service = QualityAssessmentService(make_extraction_settings(**settings))
    return service.assess(result or confident_result(data), today=TODAY)


def test_clean_invoice_passes() -> None:
    report = assess(invoice())
    assert report.quality_status is QualityStatus.PASS
    assert report.review_required is False
    assert report.quality_score == 100
    assert report.issue_counts.total == 0
    assert report.field_review == ()
    assert report.validation_summary.schema_valid is True


def test_status_and_score_are_not_accuracy_claims() -> None:
    report = assess(invoice())
    assert "not a measure of ai accuracy" in report.assessment_summary.lower()
    assert "not ai accuracy" in report.score_explanation.lower()


def test_low_confidence_important_field_flags_review() -> None:
    result = make_result(
        invoice(),
        field_confidence={
            "vendor_name": FieldConfidenceEntry(value="Acme Supplies", confidence=0.4, evidence="blurry"),
            "invoice_number": FieldConfidenceEntry(value="INV-1", confidence=0.95),
            "invoice_date": FieldConfidenceEntry(value=date(2026, 1, 2), confidence=0.95),
            "currency": FieldConfidenceEntry(value="USD", confidence=0.95),
            "total": FieldConfidenceEntry(value=Decimal("11.00"), confidence=0.95),
        },
    )
    report = assess(invoice(), result)
    assert report.quality_status is QualityStatus.REVIEW_REQUIRED
    assert report.review_required is True
    assert any(issue.rule_id == "CONF-002" for issue in report.issues)
    reviews = {review.field_path: review for review in report.field_review}
    assert reviews["vendor_name"].confidence == 0.4
    assert reviews["vendor_name"].evidence == "blurry"
    assert reviews["vendor_name"].value == "Acme Supplies"


def test_missing_confidence_metadata_is_flagged() -> None:
    report = assess(invoice(), make_result(invoice()))
    assert any(issue.rule_id == "CONF-001" for issue in report.issues)
    assert any(not review.has_confidence_metadata for review in report.field_review)
    assert report.quality_status is QualityStatus.REVIEW_REQUIRED


def test_missing_critical_value_yields_incomplete() -> None:
    report = assess(invoice(total=None), confident_result(invoice(total=None)))
    assert report.quality_status is QualityStatus.INCOMPLETE
    assert any(issue.rule_id == "GEN-007" for issue in report.issues)


def test_failed_extraction_yields_failed() -> None:
    result = make_result(
        None,
        document_type=DocumentType.INVOICE,
        status=ProcessingStatus.FAILED,
        validation_status=ValidationStatus.INVALID,
    )
    report = assess(invoice(), result)
    assert report.quality_status is QualityStatus.FAILED
    assert report.review_required is True
    assert any(issue.rule_id == "GEN-001" for issue in report.issues)


def test_schema_validation_failure_yields_failed() -> None:
    result = make_result(
        invoice(),
        validation_status=ValidationStatus.INVALID,
        validation_errors=[
            ErrorDetail(code="value_error", message="total is invalid", field_path="total")
        ],
    )
    report = assess(invoice(), result)
    assert report.quality_status is QualityStatus.FAILED
    assert report.validation_summary.schema_valid is False
    assert any(issue.rule_id == "GEN-003" for issue in report.issues)
    assert report.issue_counts.critical >= 1


def test_empty_extracted_content_is_flagged() -> None:
    empty = invoice(
        vendor_name=None,
        invoice_number=None,
        invoice_date=None,
        due_date=None,
        currency=None,
        subtotal=None,
        tax=None,
        total=None,
        line_items=[],
    )
    report = assess(empty, confident_result(empty))
    assert any(issue.rule_id == "GEN-006" for issue in report.issues)
    assert report.quality_status is QualityStatus.INCOMPLETE


def test_unreadable_content_is_flagged() -> None:
    data = invoice(unreadable_fields=["vendor_name"])
    report = assess(data, confident_result(data))
    assert any(issue.rule_id == "GEN-008" for issue in report.issues)
    assert report.validation_summary.unreadable_fields == ("vendor_name",)


def test_generic_warning_does_not_become_error() -> None:
    data = invoice()
    result = confident_result(data)
    result.warnings.append("Confidence values are model estimates, not calibrated probabilities.")
    result.warnings.append("A noncritical extraction note.")
    report = assess(data, result)
    assert report.quality_status is QualityStatus.PASS
    assert any(issue.rule_id == "GEN-012" for issue in report.issues)
    assert report.issue_counts.info >= 1
    assert report.issue_counts.error == 0


def test_detected_type_mismatch_is_flagged() -> None:
    service = QualityAssessmentService(make_extraction_settings())
    result = confident_result(invoice())
    report = service.assess(result, detected_type=DocumentType.RECEIPT, today=TODAY)
    assert any(issue.rule_id == "GEN-013" for issue in report.issues)
    assert report.quality_status is QualityStatus.REVIEW_REQUIRED


def test_partial_pages_yield_incomplete() -> None:
    settings = make_extraction_settings()
    _, record = ingest_upload(FakeUpload("multi.pdf", make_pdf(pages=2), "file-1"), settings)
    assert record is not None
    processed = preprocess_document(record, settings)
    failed_page = processed.pages[1].model_copy(
        update={"processing_status": ProcessingStatus.FAILED, "error": "page failed"}
    )
    processed = processed.model_copy(update={"pages": (processed.pages[0], failed_page)})
    result = confident_result(invoice())
    result.source_document_id = processed.document_id
    report = QualityAssessmentService(settings).assess(result, preprocessing=processed, today=TODAY)
    assert report.quality_status is QualityStatus.INCOMPLETE
    assert report.validation_summary.partial_page_processing is True
    assert report.validation_summary.failed_pages == 1
    assert any(issue.rule_id == "GEN-010" for issue in report.issues)


def test_issue_limit_is_bounded_and_deterministic() -> None:
    bad_lines = [
        InvoiceLineItem(
            description=f"Item {index}",
            quantity=Decimal("2"),
            unit_price=Decimal("5"),
            amount=Decimal("999"),
        )
        for index in range(10)
    ]
    data = invoice(line_items=bad_lines, subtotal=Decimal("9990"), total=Decimal("9990"))
    result = confident_result(data)
    first = assess(data, result, MAX_QUALITY_ISSUES="3")
    second = assess(data, result, MAX_QUALITY_ISSUES="3")
    assert len(first.issues) == 3
    assert [issue.issue_id for issue in first.issues] == [issue.issue_id for issue in second.issues]
    assert [issue.severity for issue in first.issues] == [
        QualitySeverity.ERROR,
        QualitySeverity.ERROR,
        QualitySeverity.ERROR,
    ]


def test_score_deductions_are_transparent() -> None:
    issues = (
        QualityIssue(
            issue_id="q1",
            rule_id="X",
            severity=QualitySeverity.CRITICAL,
            category=QualityCategory.SCHEMA,
            description="critical",
            recommended_action="fix",
        ),
        QualityIssue(
            issue_id="q2",
            rule_id="Y",
            severity=QualitySeverity.WARNING,
            category=QualityCategory.REVIEW,
            description="warning",
            recommended_action="review",
        ),
    )
    assert calculate_quality_score(issues) == 55
