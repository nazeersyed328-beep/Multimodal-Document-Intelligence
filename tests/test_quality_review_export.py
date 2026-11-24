"""Tests for session review state and safe exports."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

from src.extraction.export import dump_extraction_result
from src.quality import (
    QualityAssessmentService,
    add_reviewer_note,
    apply_review_to_report,
    build_reviewed_output,
    dump_original_extraction,
    dump_quality_report,
    dump_reviewed_output,
    new_review_state,
    render_quality_markdown,
    reset_review,
    review_progress,
    set_field_edit,
    set_issue_review,
)
from src.schemas.common import FieldConfidenceEntry
from src.schemas.extraction_result import ExtractionResult
from src.schemas.invoice import InvoiceDocument
from src.schemas.quality import IssueReviewStatus, QualityReport, ReviewState

from .conftest import make_extraction_settings, make_result

TODAY = date(2026, 1, 15)
NOW = datetime(2026, 1, 3, tzinfo=timezone.utc)


def build_fixture() -> tuple[ExtractionResult, QualityReport]:
    data = InvoiceDocument(
        document_id="doc-review",
        source_filename="invoice.png",
        extracted_at=NOW,
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
        field_confidence={
            "vendor_name": FieldConfidenceEntry(value="Acme", confidence=0.4, evidence="blurry"),
            "invoice_number": FieldConfidenceEntry(value="INV-1", confidence=0.95),
            "invoice_date": FieldConfidenceEntry(value=date(2026, 1, 2), confidence=0.95),
            "currency": FieldConfidenceEntry(value="USD", confidence=0.95),
            "total": FieldConfidenceEntry(value=Decimal("11.00"), confidence=0.95),
        },
    )
    report = QualityAssessmentService(make_extraction_settings()).assess(result, today=TODAY)
    return result, report


def test_review_state_helpers_are_immutable() -> None:
    state = new_review_state("result-test")
    reviewed = set_issue_review(state, "issue-1", IssueReviewStatus.RESOLVED, "verified")
    assert state.issue_reviews == {}
    assert reviewed.issue_reviews["issue-1"].status is IssueReviewStatus.RESOLVED
    noted = add_reviewer_note(reviewed, "checked source")
    assert reviewed.reviewer_notes == ()
    assert noted.reviewer_notes == ("checked source",)
    assert reset_review(noted) == new_review_state("result-test") or reset_review(noted).result_id == "result-test"


def test_field_edit_does_not_mutate_original_result() -> None:
    result, report = build_fixture()
    state = new_review_state(result.result_id)
    state = set_field_edit(
        state,
        "vendor_name",
        "Acme Corp",
        "verified from source",
        original_value=result.data.vendor_name,
    )
    output = build_reviewed_output(result, report, state)
    assert result.data.vendor_name == "Acme"
    assert output.data["vendor_name"] == "Acme Corp"
    assert output.changed_fields == ("vendor_name",)
    assert output.field_edits[0].original_value == "Acme"
    assert output.verified_against_source is False
    assert "not verified against" in output.verification_note.lower()


def test_field_edit_creates_missing_parent_containers() -> None:
    result, report = build_fixture()
    state = set_field_edit(
        new_review_state(result.result_id),
        "line_items[0].description",
        "Widget",
        original_value=None,
    )
    output = build_reviewed_output(result, report, state)
    assert output.data["line_items"][0]["description"] == "Widget"
    assert output.changed_fields == ("line_items[0].description",)


def test_resolved_review_keeps_detected_issue() -> None:
    _, report = build_fixture()
    issue_id = report.issues[0].issue_id
    state = set_issue_review(new_review_state(report.result_id), issue_id, IssueReviewStatus.RESOLVED)
    updated = apply_review_to_report(report, state)
    assert len(updated.issues) == len(report.issues)
    assert next(issue for issue in updated.issues if issue.issue_id == issue_id).resolved is True
    assert next(issue for issue in report.issues if issue.issue_id == issue_id).resolved is False
    assert review_progress(updated, state)["resolved"] == 1


def test_reset_review_clears_all_changes() -> None:
    result, _ = build_fixture()
    state = new_review_state(result.result_id)
    state = set_field_edit(state, "vendor_name", "New", original_value="Acme")
    state = set_issue_review(state, "issue-1", IssueReviewStatus.REVIEWED)
    state = add_reviewer_note(state, "note")
    assert reset_review(state).field_edits == {}
    assert reset_review(state).issue_reviews == {}
    assert reset_review(state).reviewer_notes == ()


def test_quality_json_export_is_parseable_and_includes_review() -> None:
    result, report = build_fixture()
    state: ReviewState = set_issue_review(
        new_review_state(result.result_id), report.issues[0].issue_id, IssueReviewStatus.REVIEWED
    )
    payload = json.loads(dump_quality_report(report, state))
    assert payload["schema_version"] == "4.0"
    assert payload["quality_assessment"]["result_id"] == result.result_id
    assert payload["review_state"]["result_id"] == result.result_id
    assert "image_bytes" not in json.dumps(payload)


def test_reviewed_json_export_is_parseable_and_labeled() -> None:
    result, report = build_fixture()
    state = new_review_state(result.result_id)
    output = build_reviewed_output(result, report, state)
    payload = json.loads(dump_reviewed_output(output))
    assert payload["schema_version"] == "4.0"
    assert payload["reviewed_output"]["output_type"] == "reviewed_output"
    assert payload["reviewed_output"]["verified_against_source"] is False


def test_markdown_report_includes_quality_and_safety_notices() -> None:
    result, report = build_fixture()
    state = new_review_state(result.result_id)
    output = build_reviewed_output(result, report, state)
    markdown = render_quality_markdown(report, result, state, output)
    assert "# Extraction quality report" in markdown
    assert "not AI accuracy" in markdown
    assert "model estimates, not calibrated probabilities" in markdown
    assert "immutable" in markdown
    assert "CONF-002" in markdown
    assert "not verified against the source" in markdown


def test_original_extraction_export_matches_existing_exporter() -> None:
    result, _ = build_fixture()
    assert dump_original_extraction(result) == dump_extraction_result(result)
