"""Phase 5 evaluation package."""

from .comparison import compare_values, evaluate_fields, flatten_payload, normalize_string
from .dataset import (
    DEFAULT_EVALUATION_ROOT,
    EvaluationSample,
    load_dataset,
    load_ground_truths,
    load_index,
    sample_version,
    select_samples,
    validate_dataset,
)
from .fixtures import FixtureVisionProvider, unflatten_fields
from .metrics import (
    build_error_analysis,
    compute_document_type_metrics,
    compute_field_metrics,
    compute_field_metrics_from_comparisons,
    compute_per_document_type_metrics,
    compute_performance,
    compute_quality_metrics,
    compute_schema_validity,
)
from .reporting import render_markdown_report, write_reports
from .runner import EvaluationRunner
from .schemas import (
    ComparisonStatus,
    DocumentEvaluation,
    ErrorEntry,
    EvaluationMode,
    EvaluationRunSummary,
    EvaluationSubset,
    FieldComparison,
    GroundTruth,
)

__all__ = [
    "ComparisonStatus",
    "DEFAULT_EVALUATION_ROOT",
    "DocumentEvaluation",
    "ErrorEntry",
    "EvaluationMode",
    "EvaluationRunSummary",
    "EvaluationRunner",
    "EvaluationSample",
    "EvaluationSubset",
    "FieldComparison",
    "FixtureVisionProvider",
    "GroundTruth",
    "build_error_analysis",
    "compare_values",
    "compute_document_type_metrics",
    "compute_field_metrics",
    "compute_field_metrics_from_comparisons",
    "compute_per_document_type_metrics",
    "compute_performance",
    "compute_quality_metrics",
    "compute_schema_validity",
    "evaluate_fields",
    "flatten_payload",
    "load_dataset",
    "load_ground_truths",
    "load_index",
    "normalize_string",
    "render_markdown_report",
    "sample_version",
    "select_samples",
    "unflatten_fields",
    "validate_dataset",
    "write_reports",
]
