"""Streamlit AppTest checks for the final application interface."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import get_settings
from src.document.ingestion import ingest_upload
from src.document.preprocessing import preprocess_document
from src.document.session import DocumentSession
from src.schemas.common import FieldConfidenceEntry
from src.schemas.invoice import InvoiceDocument

from .conftest import (
    FakeUpload,
    make_extraction_settings,
    make_png,
    make_result,
)

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_renders_without_crashing() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert app.title[0].value == "Multimodal Document Intelligence"


def test_app_labels_phase_and_upload_section() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    captions = [item.value for item in app.caption]
    assert any("Phase 5" in caption for caption in captions)
    assert any("Accepted:" in caption for caption in captions)
    headers = [item.value for item in app.subheader]
    assert "1. Upload documents" in headers
    assert "2. Document list" in headers
    assert "3. Processing controls" in headers
    assert "6. AI extraction" in headers
    assert "8. Evaluation" in headers


def test_app_reports_no_documents_initially() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert any("No validated documents yet" in item.value for item in app.info)


def test_preview_handles_stale_page_selection(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    get_settings.cache_clear()
    try:
        settings = make_extraction_settings()
        _, record = ingest_upload(FakeUpload("invoice.png", make_png(), "file-preview"), settings)
        assert record is not None
        session = DocumentSession()
        session.add_record(record)
        session.set_processed(record.document_id, preprocess_document(record, settings))
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.session_state["phase2_document_session"] = session
        app.session_state["preview_page_select"] = 999
        app.run()
        assert not app.exception, [str(item.value) for item in app.exception]
        headers = [item.value for item in app.subheader]
        assert "4. Preprocessing status" in headers
        assert "5. Preview" in headers
    finally:
        get_settings.cache_clear()


def test_app_renders_quality_section_and_exports(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    get_settings.cache_clear()
    try:
        settings = make_extraction_settings()
        _, record = ingest_upload(FakeUpload("invoice.png", make_png(), "file-quality"), settings)
        assert record is not None
        session = DocumentSession()
        session.add_record(record)
        session.set_processed(record.document_id, preprocess_document(record, settings))
        data = InvoiceDocument(
            document_id=record.document_id,
            source_filename=record.sanitized_filename,
            extracted_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            vendor_name="Acme",
            invoice_number="INV-1",
            invoice_date=date(2026, 1, 2),
            due_date=date(2026, 1, 20),
            currency="USD",
            subtotal=Decimal("10.00"),
            tax=Decimal("1.00"),
            total=Decimal("11.00"),
        )
        result = make_result(
            data,
            result_id="result-quality",
            source_document_id=record.document_id,
            field_confidence={
                "vendor_name": FieldConfidenceEntry(value="Acme", confidence=0.4, evidence="blurry"),
                "invoice_number": FieldConfidenceEntry(value="INV-1", confidence=0.95),
                "invoice_date": FieldConfidenceEntry(value=date(2026, 1, 2), confidence=0.95),
                "currency": FieldConfidenceEntry(value="USD", confidence=0.95),
                "total": FieldConfidenceEntry(value=Decimal("11.00"), confidence=0.95),
            },
        )
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.session_state["phase2_document_session"] = session
        app.session_state["phase3_extraction_results"] = {record.document_id: result}
        app.run()
        assert not app.exception, [str(item.value) for item in app.exception]
        assert any("7. Extraction quality" in item.value for item in app.subheader)
        assert any("not AI accuracy" in item.value for item in app.caption)
        assert any("Review low-confidence fields" in item.value for item in app.markdown)
        assert any("Quality and review exports" in item.value for item in app.markdown)
    finally:
        get_settings.cache_clear()
