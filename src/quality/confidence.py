"""Field-level confidence analysis using only model-supplied metadata."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.schemas.common import DocumentType, FieldConfidenceEntry
from src.schemas.extraction_result import ExtractionResult
from src.schemas.quality import (
    FieldReview,
    QualityCategory,
    QualityIssue,
    QualitySeverity,
)

from .rules import make_issue

IMPORTANT_FIELDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.INVOICE: (
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "currency",
        "total",
    ),
    DocumentType.RECEIPT: (
        "merchant_name",
        "transaction_datetime",
        "currency",
        "total",
    ),
    DocumentType.HANDWRITTEN_NOTE: ("transcribed_text", "language"),
    DocumentType.DIAGRAM: ("title", "elements"),
}

_PATH_PARTS = re.compile(r"[^.\[\]]+")


def _lookup_value(payload: Any, field_path: str) -> Any:
    if payload is None:
        return None
    value: Any = payload.model_dump(mode="python") if hasattr(payload, "model_dump") else payload
    for part in _PATH_PARTS.findall(field_path):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, (list, tuple)) and part.isdigit():
            index = int(part)
            value = value[index] if 0 <= index < len(value) else None
        else:
            return None
        if value is None:
            return None
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def build_field_reviews(
    result: ExtractionResult,
    threshold: float,
) -> tuple[FieldReview, ...]:
    """Build a review list from existing confidence metadata only."""
    important = set(IMPORTANT_FIELDS.get(result.document_type, ()))
    reviews: list[FieldReview] = []

    for field_path, entry in result.field_confidence.items():
        confidence = entry.confidence
        has_metadata = confidence is not None or bool(entry.evidence)
        reasons: list[str] = []
        severity = QualitySeverity.WARNING

        if confidence is None:
            reasons.append("No confidence score was supplied for this field.")
        elif confidence < threshold:
            reasons.append(
                f"Confidence {confidence:.2f} is below the review threshold {threshold:.2f}."
            )
        if entry.needs_review:
            reasons.append("The extraction marked this field for review.")
        value = _lookup_value(result.data, field_path)
        if _is_empty(value):
            reasons.append("The extracted value is missing or empty.")
        if not has_metadata and field_path in important:
            reasons.append("This important field has no confidence metadata.")
            severity = QualitySeverity.ERROR

        if not reasons:
            continue
        if field_path in important and (confidence is None or (confidence is not None and confidence < threshold)):
            severity = QualitySeverity.ERROR

        reviews.append(
            FieldReview(
                field_path=field_path,
                value=entry.value if entry.value is not None else value,  # type: ignore[arg-type]
                confidence=confidence,
                evidence=entry.evidence,
                needs_review=True,
                has_confidence_metadata=has_metadata,
                severity=severity,
                reason=" ".join(reasons),
                recommended_action="Compare this value with the source document before accepting it.",
            )
        )

    for field_path in IMPORTANT_FIELDS.get(result.document_type, ()):
        if field_path in result.field_confidence:
            continue
        value = _lookup_value(result.data, field_path)
        reviews.append(
            FieldReview(
                field_path=field_path,
                value=value,  # type: ignore[arg-type]
                confidence=None,
                evidence=None,
                needs_review=True,
                has_confidence_metadata=False,
                severity=QualitySeverity.WARNING,
                reason=(
                    "No confidence metadata was supplied for this important field. "
                    + ("The extracted value is also missing or empty." if _is_empty(value) else "")
                ),
                recommended_action="Review the field and do not infer a confidence score.",
            )
        )
    return tuple(reviews)


def build_confidence_issues(
    result: ExtractionResult,
    threshold: float,
    field_reviews: tuple[FieldReview, ...],
) -> list[QualityIssue]:
    """Create issues for confidence gaps without inventing new scores."""
    issues: list[QualityIssue] = []
    if result.data is not None and not result.field_confidence:
        issues.append(
            make_issue(
                "CONF-001",
                QualitySeverity.WARNING,
                QualityCategory.CONFIDENCE,
                "No field-level confidence metadata was supplied for a successful extraction.",
                recommended_action="Review extracted values directly; no confidence score is available.",
            )
        )

    for review in field_reviews:
        if review.severity is QualitySeverity.ERROR:
            issues.append(
                make_issue(
                    "CONF-002",
                    QualitySeverity.ERROR,
                    QualityCategory.CONFIDENCE,
                    f"Important field {review.field_path} is low-confidence or lacks usable confidence metadata.",
                    affected_fields=(review.field_path,),
                    expected_value=f">= {threshold:.2f}" if review.confidence is not None else "confidence metadata",
                    actual_value=review.confidence,
                    recommended_action=review.recommended_action,
                )
            )

    if result.overall_confidence is not None and result.overall_confidence < threshold:
        issues.append(
            make_issue(
                "CONF-003",
                QualitySeverity.WARNING,
                QualityCategory.CONFIDENCE,
                "Overall model-estimated confidence is below the configured review threshold.",
                affected_fields=("overall",),
                expected_value=f">= {threshold:.2f}",
                actual_value=result.overall_confidence,
                recommended_action="Treat this extraction as model-estimated and review it before use.",
            )
        )
    return issues
