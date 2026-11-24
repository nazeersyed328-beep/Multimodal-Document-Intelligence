"""Schemas for Phase 5 evaluation datasets, results, and reports."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import Field

from src.schemas.common import DocumentType, StrictModel


class EvaluationMode(str, Enum):
    """Clear provenance label for every evaluation report."""

    LOCAL_SYNTHETIC_TESTS = "local_synthetic_tests"
    MOCKED_API_TESTS = "mocked_api_tests"
    REAL_GEMINI_API_EVALUATION = "real_gemini_api_evaluation"


class EvaluationSubset(str, Enum):
    SMALL = "small"
    FULL = "full"


class ComparisonStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    MISSING = "missing"
    EXTRA = "extra"
    UNCERTAIN = "uncertain"


class GroundTruth(StrictModel):
    """Manually authored expectations for one synthetic sample."""

    schema_version: str = "5.0"
    sample_id: str = Field(min_length=1, max_length=160)
    ground_truth_id: str = Field(min_length=1, max_length=160)
    document_type: DocumentType
    expected_document_type: DocumentType
    file_name: str = Field(min_length=1, max_length=300)
    dataset_subset: EvaluationSubset = EvaluationSubset.FULL
    dataset_version: str = Field(min_length=1, max_length=100)
    tags: tuple[str, ...] = ()
    source: str = Field(min_length=1, max_length=1000)
    license: str = Field(min_length=1, max_length=200)
    checked_by: str = Field(min_length=1, max_length=300)
    checked_at: date
    expected_fields: dict[str, Any]
    expected_issue_rules: tuple[str, ...] = ()
    expected_review_required: bool | None = None
    expected_low_confidence_fields: tuple[str, ...] = ()
    expected_unreadable_fields: tuple[str, ...] = ()
    notes: str = ""


class FieldComparison(StrictModel):
    """One field-level ground-truth comparison."""

    field_path: str
    expected_value: Any = None
    extracted_value: Any = None
    status: ComparisonStatus
    exact_match: bool
    normalized_match: bool
    comparison_method: str
    uncertain: bool = False
    confidence: float | None = None
    note: str = ""


class ErrorEntry(StrictModel):
    """One error-analysis record without private document content."""

    sample_id: str
    document_type: str
    field_path: str
    expected_value: Any = None
    actual_value: Any = None
    error_category: str
    flagged_for_review: bool | None = None
    potential_cause: str = ""


class DocumentEvaluation(StrictModel):
    """Local evaluation result for one sample."""

    run_id: str
    sample_id: str
    document_type: str
    expected_document_type: str
    extraction_status: str
    schema_valid: bool
    mode: EvaluationMode
    field_metrics: dict[str, Any]
    comparisons: tuple[FieldComparison, ...] = ()
    detected_issue_rules: tuple[str, ...] = ()
    expected_issue_rules: tuple[str, ...] = ()
    flagged_low_confidence_fields: tuple[str, ...] = ()
    flagged_unreadable_fields: tuple[str, ...] = ()
    quality_review_required: bool | None = None
    expected_review_required: bool | None = None
    latency_seconds: float
    provider_requests: int
    retry_count: int
    error_code: str | None = None
    error_message: str | None = None
    fingerprint: str


class EvaluationRunSummary(StrictModel):
    """Complete benchmark summary for one run."""

    run_id: str
    generated_at: datetime
    mode: EvaluationMode
    subset: EvaluationSubset
    dataset_version: str
    model_id: str
    api_status: str
    sample_count: int
    schema_validity: dict[str, Any]
    field_metrics: dict[str, Any]
    document_type_metrics: dict[str, Any]
    per_document_type_metrics: dict[str, Any]
    quality_metrics: dict[str, Any]
    performance: dict[str, Any]
    document_evaluations: tuple[DocumentEvaluation, ...] = ()
    error_analysis: tuple[ErrorEntry, ...] = ()
    limitations: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    raw_results_directory: str = ""
    resumed_count: int = 0
    dataset_validation: dict[str, Any] = Field(default_factory=dict)
