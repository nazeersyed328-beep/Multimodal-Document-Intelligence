"""Structured schemas for Phase 4 quality assessment and human review."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from .common import (
    DocumentType,
    ErrorDetail,
    StrictModel,
    ValidationStatus,
)
from .extraction_result import ProcessingStatus


class QualityStatus(str, Enum):
    """Rule-based quality outcome; not a measure of AI accuracy."""

    PASS = "pass"
    REVIEW_REQUIRED = "review_required"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class QualitySeverity(str, Enum):
    """Severity of a deterministic quality finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityCategory(str, Enum):
    """Category used to group and filter quality findings."""

    SCHEMA = "schema"
    CONFIDENCE = "confidence"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    PROCESSING = "processing"
    DOCUMENT_TYPE = "document_type"
    CONTENT = "content"
    CURRENCY = "currency"
    DATE = "date"
    REVIEW = "review"


QualityValue = str | int | float | Decimal | bool | date | datetime | list[Any] | dict[str, Any] | None


class QualityIssue(StrictModel):
    """One deterministic finding about an extraction."""

    issue_id: str = Field(min_length=1, max_length=128)
    rule_id: str = Field(min_length=1, max_length=64)
    severity: QualitySeverity
    category: QualityCategory
    description: str = Field(min_length=1, max_length=2000)
    affected_fields: tuple[str, ...] = ()
    page_number: int | None = Field(default=None, ge=1)
    expected_value: QualityValue = None
    actual_value: QualityValue = None
    recommended_action: str = Field(min_length=1, max_length=1000)
    resolved: bool = False


class FieldReview(StrictModel):
    """A field that a human should inspect, with preserved evidence."""

    field_path: str = Field(min_length=1, max_length=500)
    value: QualityValue = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = Field(default=None, max_length=2000)
    needs_review: bool = True
    has_confidence_metadata: bool = True
    severity: QualitySeverity = QualitySeverity.WARNING
    reason: str = Field(min_length=1, max_length=1000)
    recommended_action: str = Field(min_length=1, max_length=1000)


class IssueCounts(StrictModel):
    """Issue totals by severity."""

    info: int = 0
    warning: int = 0
    error: int = 0
    critical: int = 0

    @property
    def total(self) -> int:
        return self.info + self.warning + self.error + self.critical


class ValidationSummary(StrictModel):
    """Security-safe summary of structural and preprocessing outcomes."""

    schema_valid: bool
    extraction_status: ProcessingStatus
    validation_status: ValidationStatus
    validation_errors: tuple[ErrorDetail, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    unreadable_fields: tuple[str, ...] = ()
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    partial_page_processing: bool = False


class QualityReport(StrictModel):
    """Rule-based assessment of one extraction result."""

    report_id: str = Field(min_length=1, max_length=160)
    document_id: str = Field(min_length=1, max_length=160)
    result_id: str = Field(min_length=1, max_length=160)
    document_type: DocumentType
    quality_status: QualityStatus
    review_required: bool
    quality_score: int = Field(ge=0, le=100)
    score_explanation: str = Field(min_length=1, max_length=1000)
    issue_counts: IssueCounts
    issues: tuple[QualityIssue, ...] = ()
    field_review: tuple[FieldReview, ...] = ()
    validation_summary: ValidationSummary
    assessment_summary: str = Field(min_length=1, max_length=3000)
    assessment_timestamp: datetime
    rule_version: str = "4.0"


class IssueReviewStatus(str, Enum):
    """Reviewer action recorded against a detected issue."""

    REVIEWED = "reviewed"
    RESOLVED = "resolved"


class IssueReview(StrictModel):
    """A reviewer action for one already-detected issue."""

    issue_id: str = Field(min_length=1, max_length=128)
    status: IssueReviewStatus
    note: str | None = Field(default=None, max_length=2000)
    reviewed_at: datetime


class FieldEdit(StrictModel):
    """A reviewer-supplied value that never overwrites the AI extraction."""

    field_path: str = Field(min_length=1, max_length=500)
    original_value: QualityValue = None
    edited_value: QualityValue = None
    note: str | None = Field(default=None, max_length=2000)
    edited_at: datetime


class ReviewState(StrictModel):
    """Session-only review state for one extraction result."""

    result_id: str = Field(min_length=1, max_length=160)
    issue_reviews: dict[str, IssueReview] = Field(default_factory=dict)
    field_edits: dict[str, FieldEdit] = Field(default_factory=dict)
    reviewer_notes: tuple[str, ...] = ()
    updated_at: datetime


class ReviewedOutput(StrictModel):
    """Export view that clearly distinguishes reviewer edits from AI output."""

    output_type: Literal["reviewed_output"] = "reviewed_output"
    report_id: str = Field(min_length=1, max_length=160)
    result_id: str = Field(min_length=1, max_length=160)
    document_id: str = Field(min_length=1, max_length=160)
    document_type: DocumentType
    data: dict[str, Any]
    changed_fields: tuple[str, ...] = ()
    field_edits: tuple[FieldEdit, ...] = ()
    reviewer_notes: tuple[str, ...] = ()
    verified_against_source: bool = False
    verification_note: str = (
        "Reviewed output contains session-only edits. It is not verified against "
        "the source unless a reviewer explicitly confirms each field."
    )
    generated_at: datetime
