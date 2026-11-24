"""Rule-based quality assessment for validated extraction results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from src.config import Settings
from src.schemas.common import DocumentType, ValidationStatus
from src.schemas.document_ingestion import PreprocessingResult, ProcessedPage
from src.schemas.extraction_result import ExtractionResult, ProcessingStatus
from src.schemas.handwritten_note import HandwrittenNoteDocument
from src.schemas.invoice import InvoiceDocument
from src.schemas.quality import (
    FieldReview,
    QualityCategory,
    QualityIssue,
    QualityReport,
    QualitySeverity,
    QualityStatus,
    ValidationSummary,
)
from src.schemas.receipt import ReceiptDocument

from .confidence import build_confidence_issues, build_field_reviews
from .consistency import (
    collect_diagram_issues,
    collect_invoice_issues,
    collect_note_issues,
    collect_receipt_issues,
)
from .rules import (
    SCORE_EXPLANATION,
    calculate_quality_score,
    count_by_severity,
    limit_issues,
    make_issue,
)

_METADATA_KEYS = {
    "document_id",
    "document_type",
    "source_filename",
    "extracted_at",
    "validation_status",
    "warnings",
    "errors",
    "missing_fields",
    "unreadable_fields",
}


def _lookup_value(payload: Any, field_path: str) -> Any:
    if payload is None:
        return None
    value: Any = payload.model_dump(mode="python") if hasattr(payload, "model_dump") else payload
    for part in field_path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _document_fields(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    dumped = payload.model_dump(mode="python")
    return {key: value for key, value in dumped.items() if key not in _METADATA_KEYS}


def _critical_fields(document_type: DocumentType) -> tuple[str, ...]:
    return {
        DocumentType.INVOICE: ("vendor_name", "total"),
        DocumentType.RECEIPT: ("merchant_name", "total"),
        DocumentType.HANDWRITTEN_NOTE: ("transcribed_text",),
        DocumentType.DIAGRAM: ("elements",),
    }.get(document_type, ())


def _page_summary(preprocessing: PreprocessingResult | None) -> dict[str, int | bool]:
    pages: tuple[ProcessedPage, ...] = preprocessing.pages if preprocessing else ()
    successful = sum(
        page.processing_status is ProcessingStatus.COMPLETED and bool(page.image_bytes)
        for page in pages
    )
    failed = sum(
        page.processing_status is ProcessingStatus.FAILED or bool(page.error)
        for page in pages
    )
    total = len(pages)
    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "partial": preprocessing is not None and (failed > 0 or (0 < successful < total)),
    }


def _general_issues(
    result: ExtractionResult,
    preprocessing: PreprocessingResult | None,
    detected_type: DocumentType | None,
    page_counts: Mapping[str, int | bool],
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []

    if result.status is ProcessingStatus.FAILED:
        issues.append(
            make_issue(
                "GEN-001",
                QualitySeverity.CRITICAL,
                QualityCategory.PROCESSING,
                "Extraction failed and did not produce a reliable result.",
                recommended_action="Review the extraction error before using any output.",
            )
        )

    malformed_codes = {"empty_response", "invalid_json", "invalid_response", "malformed_output"}
    if any(error.code in malformed_codes for error in result.validation_errors):
        issues.append(
            make_issue(
                "GEN-002",
                QualitySeverity.CRITICAL,
                QualityCategory.SCHEMA,
                "The model response was empty, malformed, or unusable.",
                affected_fields=tuple(
                    error.field_path for error in result.validation_errors if error.field_path
                ),
                recommended_action="Retry extraction or inspect the provider response handling.",
            )
        )

    for error in result.validation_errors:
        if error.code in malformed_codes:
            continue
        issues.append(
            make_issue(
                "GEN-003",
                QualitySeverity.CRITICAL,
                QualityCategory.SCHEMA,
                f"Schema validation failed: {error.message}",
                affected_fields=(error.field_path,) if error.field_path else (),
                actual_value=error.code,
                recommended_action="Correct the extraction output before using it.",
            )
        )

    if result.document_type is DocumentType.UNKNOWN:
        issues.append(
            make_issue(
                "GEN-004",
                QualitySeverity.ERROR,
                QualityCategory.DOCUMENT_TYPE,
                "Document type is unknown.",
                affected_fields=("document_type",),
                recommended_action="Select or confirm a supported document type before extraction.",
            )
        )

    if result.data is None and result.status is not ProcessingStatus.FAILED:
        issues.append(
            make_issue(
                "GEN-005",
                QualitySeverity.CRITICAL,
                QualityCategory.CONTENT,
                "No typed extraction payload is available.",
                recommended_action="Do not use this result until extraction produces a typed payload.",
            )
        )

    if result.data is not None and not any(
        not _is_empty(value) for value in _document_fields(result.data).values()
    ):
        issues.append(
            make_issue(
                "GEN-006",
                QualitySeverity.ERROR,
                QualityCategory.CONTENT,
                "Extracted content is empty for all document-specific fields.",
                recommended_action="Review the source document and extraction result.",
            )
        )

    missing_critical = [
        field
        for field in _critical_fields(result.document_type)
        if _is_empty(_lookup_value(result.data, field))
    ]
    if missing_critical:
        issues.append(
            make_issue(
                "GEN-007",
                QualitySeverity.ERROR,
                QualityCategory.COMPLETENESS,
                "Important information is missing from the extraction.",
                affected_fields=tuple(missing_critical),
                recommended_action="Review the missing values on the source document.",
            )
        )

    unreadable = list(result.data.unreadable_fields) if result.data is not None else []
    missing_fields = list(result.data.missing_fields) if result.data is not None else []
    if unreadable or any("unread" in field.lower() for field in missing_fields):
        issues.append(
            make_issue(
                "GEN-008",
                QualitySeverity.WARNING,
                QualityCategory.COMPLETENESS,
                "The extraction reported unreadable content.",
                affected_fields=tuple(unreadable) or ("unreadable_fields",),
                actual_value=unreadable,
                recommended_action="Inspect the unreadable regions on the source document.",
            )
        )

    if preprocessing is not None and result.source_document_id == preprocessing.document_id:
        if preprocessing.status is ProcessingStatus.FAILED:
            issues.append(
                make_issue(
                    "GEN-009",
                    QualitySeverity.CRITICAL,
                    QualityCategory.PROCESSING,
                    "Preprocessing failed for this document.",
                    recommended_action="Prepare the pages again before relying on extraction.",
                )
            )
        elif page_counts["partial"]:
            issues.append(
                make_issue(
                    "GEN-010",
                    QualitySeverity.WARNING,
                    QualityCategory.PROCESSING,
                    "Page processing was partial or contained failures.",
                    affected_fields=("processed_pages",),
                    actual_value={
                        "total_pages": page_counts["total"],
                        "successful_pages": page_counts["successful"],
                        "failed_pages": page_counts["failed"],
                    },
                    recommended_action="Review failed pages and reprocess them if needed.",
                )
            )
        for warning in preprocessing.warnings:
            issues.append(
                make_issue(
                    "GEN-011",
                    QualitySeverity.INFO,
                    QualityCategory.PROCESSING,
                    f"Preprocessing note: {warning}",
                    recommended_action="No action is required unless this affected extracted fields.",
                )
            )

    for warning in result.warnings:
        if "Confidence values are model estimates" in warning:
            continue
        issues.append(
            make_issue(
                "GEN-012",
                QualitySeverity.INFO,
                QualityCategory.REVIEW,
                f"Extraction note: {warning}",
                recommended_action="Review this note when interpreting the extraction.",
            )
        )

    if detected_type is not None and detected_type is not result.document_type:
        issues.append(
            make_issue(
                "GEN-013",
                QualitySeverity.ERROR,
                QualityCategory.DOCUMENT_TYPE,
                "Detected document type does not match the extracted document type.",
                affected_fields=("document_type",),
                expected_value=detected_type.value,
                actual_value=result.document_type.value,
                recommended_action="Confirm the document type before using the extraction.",
            )
        )

    if (
        result.status is ProcessingStatus.COMPLETED
        and result.needs_review
        and not result.validation_errors
    ):
        issues.append(
            make_issue(
                "GEN-014",
                QualitySeverity.WARNING,
                QualityCategory.REVIEW,
                "Extraction was flagged for review by the extraction stage.",
                recommended_action="Review the flagged fields and warnings before use.",
            )
        )
    return issues


def _determine_status(
    result: ExtractionResult,
    issues: tuple[QualityIssue, ...],
    field_review: tuple[FieldReview, ...],
) -> QualityStatus:
    if (
        result.status is ProcessingStatus.FAILED
        or result.data is None
        or any(issue.severity is QualitySeverity.CRITICAL for issue in issues)
    ):
        return QualityStatus.FAILED
    if any(
        issue.category in {QualityCategory.COMPLETENESS, QualityCategory.PROCESSING}
        and issue.severity in {QualitySeverity.WARNING, QualitySeverity.ERROR}
        for issue in issues
    ):
        return QualityStatus.INCOMPLETE
    if (
        result.status is ProcessingStatus.REQUIRES_REVIEW
        or result.needs_review
        or result.validation_status is ValidationStatus.NEEDS_REVIEW
        or field_review
        or any(issue.severity in {QualitySeverity.WARNING, QualitySeverity.ERROR} for issue in issues)
    ):
        return QualityStatus.REVIEW_REQUIRED
    return QualityStatus.PASS


class QualityAssessmentService:
    """Create deterministic, local-only quality reports from extraction results."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def assess(
        self,
        result: ExtractionResult,
        *,
        preprocessing: PreprocessingResult | None = None,
        detected_type: DocumentType | None = None,
        today: Any = None,
    ) -> QualityReport:
        """Assess one result. No network calls and no value modification."""
        page_counts = _page_summary(preprocessing)
        field_review = build_field_reviews(result, self.settings.confidence_review_threshold)

        issues = _general_issues(result, preprocessing, detected_type, page_counts)
        issues.extend(
            build_confidence_issues(
                result, self.settings.confidence_review_threshold, field_review
            )
        )

        payload = result.data
        if isinstance(payload, InvoiceDocument):
            issues.extend(collect_invoice_issues(payload, self.settings, today))
        elif isinstance(payload, ReceiptDocument):
            issues.extend(collect_receipt_issues(payload, self.settings, today))
        elif isinstance(payload, HandwrittenNoteDocument):
            issues.extend(
                collect_note_issues(
                    payload,
                    result.field_confidence,
                    self.settings.confidence_review_threshold,
                )
            )
        else:
            from src.schemas.diagram import DiagramDocument

            if isinstance(payload, DiagramDocument):
                issues.extend(collect_diagram_issues(payload))

        issues = limit_issues(issues, self.settings.max_quality_issues)
        report_id = f"quality-{result.result_id}"
        signed = tuple(
            issue.model_copy(update={"issue_id": f"{report_id}-Q{index:03d}"})
            for index, issue in enumerate(issues, start=1)
        )
        counts = count_by_severity(signed)
        status = _determine_status(result, signed, field_review)
        schema_valid = bool(
            payload is not None
            and not result.validation_errors
            and result.validation_status is not ValidationStatus.INVALID
        )
        validation_summary = ValidationSummary(
            schema_valid=schema_valid,
            extraction_status=result.status,
            validation_status=result.validation_status,
            validation_errors=tuple(result.validation_errors),
            warnings=tuple(result.warnings),
            missing_fields=tuple(payload.missing_fields) if payload is not None else (),
            unreadable_fields=tuple(payload.unreadable_fields) if payload is not None else (),
            total_pages=int(page_counts["total"]),
            successful_pages=int(page_counts["successful"]),
            failed_pages=int(page_counts["failed"]),
            partial_page_processing=bool(page_counts["partial"]),
        )
        score = calculate_quality_score(signed)
        summary = (
            f"Rule-based status is {status.value}. Found {counts.total} issue(s): "
            f"{counts.critical} critical, {counts.error} error, {counts.warning} warning, "
            f"{counts.info} info. {len(field_review)} field(s) require review. "
            "This is not a measure of AI accuracy."
        )
        return QualityReport(
            report_id=report_id,
            document_id=result.source_document_id or (payload.document_id if payload else "unknown"),
            result_id=result.result_id,
            document_type=result.document_type,
            quality_status=status,
            review_required=status is not QualityStatus.PASS,
            quality_score=score,
            score_explanation=SCORE_EXPLANATION,
            issue_counts=counts,
            issues=signed,
            field_review=field_review,
            validation_summary=validation_summary,
            assessment_summary=summary,
            assessment_timestamp=datetime.now(timezone.utc),
        )
