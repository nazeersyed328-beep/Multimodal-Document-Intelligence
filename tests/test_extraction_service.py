"""Tests for extraction orchestration with mocked provider responses."""

from __future__ import annotations

import pytest

from src.document.ingestion import ingest_upload
from src.document.preprocessing import preprocess_document
from src.extraction.provider import (
    MissingApiKeyError,
    QuotaExceededError,
    VisionRequest,
)
from src.extraction.service import ExtractionService
from src.schemas.common import DocumentType, ValidationStatus
from src.schemas.extraction_result import ProcessingStatus
from src.schemas.invoice import InvoiceDocument
from src.schemas.receipt import ReceiptDocument

from .conftest import (
    FakeUpload,
    FakeVisionProvider,
    make_extraction_settings,
    make_pdf,
    make_png,
    make_test_settings,
)


INVOICE_ENVELOPE = """
```json
{
  "document_type": "invoice",
  "data": {
    "vendor_name": "Acme Supplies",
    "currency": "USD",
    "invoice_number": "INV-1",
    "invoice_date": "2026-01-02",
    "subtotal": "10.00",
    "tax": "1.00",
    "total": "11.00",
    "line_items": [{"description": "Widget", "quantity": "2", "unit_price": "5.00", "amount": "10.00"}]
  },
  "missing_fields": [],
  "overall_confidence": 0.93,
  "field_confidence": [
    {"field_path": "vendor_name", "confidence": 0.95, "evidence": "Acme Supplies"},
    {"field_path": "total", "confidence": 0.95, "needs_review": false}
  ]
}
```
"""

INVOICE_DETECTION = '{"document_type":"invoice","confidence":0.96}'


def prepared(name: str, data: bytes, settings):
    _, record = ingest_upload(FakeUpload(name, data, name), settings)
    assert record is not None
    return record, preprocess_document(record, settings)


def test_explicit_invoice_extraction_with_fenced_json() -> None:
    settings = make_extraction_settings()
    record, processed = prepared("invoice.png", make_png(), settings)
    provider = FakeVisionProvider([INVOICE_ENVELOPE])
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed, DocumentType.INVOICE)

    assert result.status is ProcessingStatus.COMPLETED
    assert result.document_type is DocumentType.INVOICE
    assert isinstance(result.data, InvoiceDocument)
    assert result.data.vendor_name == "Acme Supplies"
    assert result.data.total == 11.0
    assert result.overall_confidence == 0.93
    assert result.field_confidence["vendor_name"].evidence == "Acme Supplies"
    assert result.field_confidence["total"].needs_review is False
    assert result.needs_review is False
    assert result.validation_status is ValidationStatus.VALID_WITH_WARNINGS
    assert any("not calibrated" in warning for warning in result.warnings)
    assert len(provider.requests) == 1


def test_low_confidence_field_flags_review() -> None:
    low_confidence = INVOICE_ENVELOPE.replace(
        '"confidence": 0.95, "needs_review": false',
        '"confidence": 0.4, "needs_review": false',
    )
    settings = make_extraction_settings(CONFIDENCE_REVIEW_THRESHOLD="0.7")
    record, processed = prepared("invoice.png", make_png(), settings)
    service = ExtractionService(settings, FakeVisionProvider([low_confidence]))

    result = service.extract_document(record, processed, DocumentType.INVOICE)

    assert result.field_confidence["total"].needs_review is True
    assert result.needs_review is True
    assert result.status is ProcessingStatus.REQUIRES_REVIEW


def test_auto_detection_uses_two_calls_and_respects_result() -> None:
    settings = make_extraction_settings()
    record, processed = prepared("invoice.png", make_png(), settings)
    provider = FakeVisionProvider([INVOICE_DETECTION, INVOICE_ENVELOPE])
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed)

    assert result.document_type is DocumentType.INVOICE
    assert isinstance(result.data, InvoiceDocument)
    assert len(provider.requests) == 2


def test_uncertain_detection_returns_unknown_without_extraction_call() -> None:
    settings = make_extraction_settings()
    record, processed = prepared("unknown.png", make_png(), settings)
    provider = FakeVisionProvider(
        [
            '{"document_type":"receipt","confidence":0.2,"reason":"blurry"}',
            INVOICE_ENVELOPE,
        ]
    )
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed)

    assert result.document_type is DocumentType.UNKNOWN
    assert result.status is ProcessingStatus.REQUIRES_REVIEW
    assert result.data is None
    assert result.needs_review is True
    assert len(provider.requests) == 1
    assert any("blurry" in warning for warning in result.warnings)


def test_explicit_unknown_skips_provider() -> None:
    settings = make_extraction_settings()
    record, processed = prepared("unknown.png", make_png(), settings)
    provider = FakeVisionProvider([])
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed, DocumentType.UNKNOWN)

    assert result.status is ProcessingStatus.REQUIRES_REVIEW
    assert result.data is None
    assert provider.requests == []


def test_validation_correction_retry_succeeds() -> None:
    invalid = """
    {"document_type":"invoice","data":{"vendor_name":"Acme","total":12.5}}
    """
    settings = make_extraction_settings()
    record, processed = prepared("invoice.png", make_png(), settings)
    provider = FakeVisionProvider([invalid, INVOICE_ENVELOPE])
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed, DocumentType.INVOICE)

    assert isinstance(result.data, InvoiceDocument)
    assert len(provider.requests) == 2
    second_prompt = provider.requests[1].prompt
    assert "Validation issues" in second_prompt
    assert "total" in second_prompt


def test_validation_retry_limit_returns_failure() -> None:
    settings = make_extraction_settings(MAX_RETRIES="2")
    record, processed = prepared("invoice.png", make_png(), settings)
    provider = FakeVisionProvider(
        ['{"document_type":"invoice","data":{"total":1.5}}'] * 3
    )
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed, DocumentType.INVOICE)

    assert result.status is ProcessingStatus.FAILED
    assert result.validation_status is ValidationStatus.INVALID
    assert result.data is None
    assert result.needs_review is True
    assert result.validation_errors
    assert len(provider.requests) == 3


def test_incorrect_document_type_is_corrected() -> None:
    wrong = """
    {"document_type":"receipt","data":{"merchant_name":"Shop","total":"5.00"}}
    """
    settings = make_extraction_settings()
    record, processed = prepared("invoice.png", make_png(), settings)
    provider = FakeVisionProvider([wrong, INVOICE_ENVELOPE])
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed, DocumentType.INVOICE)

    assert isinstance(result.data, InvoiceDocument)
    assert "document_type" in provider.requests[1].prompt


def test_receipt_extraction() -> None:
    response = """
    {"document_type":"receipt","data":{"merchant_name":"Corner Shop","currency":"USD","total":"8.25","transaction_datetime":"2026-02-03T10:30:00"}}
    """
    settings = make_extraction_settings()
    record, processed = prepared("receipt.png", make_png(), settings)
    service = ExtractionService(settings, FakeVisionProvider([response]))

    result = service.extract_document(record, processed, DocumentType.RECEIPT)

    assert isinstance(result.data, ReceiptDocument)
    assert result.data.merchant_name == "Corner Shop"
    assert result.data.total == 8.25


def test_handwritten_note_and_diagram_extraction() -> None:
    note = """
    {"document_type":"handwritten_note","data":{"title":"Tasks","transcribed_text":"Buy milk","unreadable_markers":[{"marker_id":"u1"}]}}
    """
    diagram = """
    {"document_type":"diagram","data":{"title":"Flow","elements":[{"element_id":"a","label":"Start"}],"ambiguities":[{"description":"方向 unclear"}]}}
    """
    settings = make_extraction_settings()
    image_record, image_processed = prepared("note.png", make_png(), settings)
    note_result = ExtractionService(
        settings, FakeVisionProvider([note])
    ).extract_document(image_record, image_processed, DocumentType.HANDWRITTEN_NOTE)
    diagram_result = ExtractionService(
        settings, FakeVisionProvider([diagram])
    ).extract_document(image_record, image_processed, DocumentType.DIAGRAM)

    assert note_result.data is not None
    assert note_result.data.document_type is DocumentType.HANDWRITTEN_NOTE
    assert diagram_result.data is not None
    assert diagram_result.data.document_type is DocumentType.DIAGRAM
    assert diagram_result.data.ambiguities[0].description == "方向 unclear"


def test_arithmetic_inconsistency_flags_review() -> None:
    response = """
    {"document_type":"invoice","data":{"subtotal":"10.00","tax":"1.00","total":"99.00"}}
    """
    settings = make_extraction_settings()
    record, processed = prepared("invoice.png", make_png(), settings)
    result = ExtractionService(
        settings, FakeVisionProvider([response])
    ).extract_document(record, processed, DocumentType.INVOICE)

    assert result.needs_review is True
    assert any("inconsistent" in warning for warning in result.warnings)


def test_missing_api_key_is_raised_lazily() -> None:
    service = ExtractionService(make_test_settings())
    with pytest.raises(MissingApiKeyError):
        _ = service.provider


def test_quota_error_becomes_failed_result() -> None:
    class RaisingProvider:
        name = "raising"

        def extract(self, request: VisionRequest):
            raise QuotaExceededError()

    settings = make_extraction_settings()
    record, processed = prepared("invoice.png", make_png(), settings)
    result = ExtractionService(settings, RaisingProvider()).extract_document(
        record, processed, DocumentType.INVOICE
    )

    assert result.status is ProcessingStatus.FAILED
    assert result.validation_errors[0].code == "quota_exceeded"
    assert "quota" in result.validation_errors[0].message.lower()
    assert "test-key-not-real" not in str(result.model_dump())


def test_multipage_pdf_sends_all_pages_in_order() -> None:
    settings = make_extraction_settings(MAX_PAGES_TO_PROCESS="3")
    record, processed = prepared("multi.pdf", make_pdf(pages=3), settings)
    provider = FakeVisionProvider([INVOICE_ENVELOPE])
    service = ExtractionService(settings, provider)

    result = service.extract_document(record, processed, DocumentType.INVOICE)

    assert isinstance(result.data, InvoiceDocument)
    assert len(provider.requests) == 1
    assert [image.page_number for image in provider.requests[0].images] == [1, 2, 3]


def test_failed_page_is_excluded_from_provider_input() -> None:
    settings = make_extraction_settings()
    record, processed = prepared("multi.pdf", make_pdf(pages=2), settings)
    failed = processed.pages[1].model_copy(
        update={"processing_status": ProcessingStatus.FAILED, "image_bytes": b""}
    )
    processed = processed.model_copy(update={"pages": (processed.pages[0], failed)})
    provider = FakeVisionProvider([INVOICE_ENVELOPE])

    ExtractionService(settings, provider).extract_document(
        record, processed, DocumentType.INVOICE
    )

    assert [image.page_number for image in provider.requests[0].images] == [1]


def test_no_prepared_pages_returns_failed_result() -> None:
    settings = make_extraction_settings()
    record, processed = prepared("bad.png", make_png(), settings)
    processed = processed.model_copy(update={"pages": ()})
    result = ExtractionService(settings, FakeVisionProvider([])).extract_document(
        record, processed, DocumentType.INVOICE
    )
    assert result.status is ProcessingStatus.FAILED
    assert result.validation_errors[0].code == "no_prepared_pages"
