"""Shared helpers for deterministic quality findings."""

from __future__ import annotations

from decimal import Decimal

from src.schemas.quality import (
    QualityCategory,
    QualityIssue,
    QualitySeverity,
)

SEVERITY_RANK = {
    QualitySeverity.CRITICAL: 4,
    QualitySeverity.ERROR: 3,
    QualitySeverity.WARNING: 2,
    QualitySeverity.INFO: 1,
}

SCORE_DEDUCTIONS = {
    QualitySeverity.CRITICAL: 40,
    QualitySeverity.ERROR: 15,
    QualitySeverity.WARNING: 5,
    QualitySeverity.INFO: 1,
}

SCORE_EXPLANATION = (
    "Rule-based quality indicator: start at 100, subtract 40 per critical, "
    "15 per error, 5 per warning, and 1 per info issue, then floor at 0. "
    "It is not AI accuracy, correctness probability, or a calibrated score."
)


def make_issue(
    rule_id: str,
    severity: QualitySeverity,
    category: QualityCategory,
    description: str,
    *,
    affected_fields: tuple[str, ...] = (),
    expected_value: object = None,
    actual_value: object = None,
    recommended_action: str,
) -> QualityIssue:
    """Create an unsigned issue; IDs are assigned by the assessment service."""
    return QualityIssue(
        issue_id="pending",
        rule_id=rule_id,
        severity=severity,
        category=category,
        description=description[:2000],
        affected_fields=affected_fields,
        expected_value=expected_value,  # type: ignore[arg-type]
        actual_value=actual_value,  # type: ignore[arg-type]
        recommended_action=recommended_action[:1000],
    )


def limit_issues(issues: list[QualityIssue], maximum: int) -> list[QualityIssue]:
    """Keep the most important findings while preserving deterministic order."""
    ordered = sorted(
        issues,
        key=lambda issue: (
            -SEVERITY_RANK[issue.severity],
            issue.rule_id,
            issue.description,
            ",".join(issue.affected_fields),
        ),
    )
    return ordered[:maximum]


def count_by_severity(issues: tuple[QualityIssue, ...]):
    """Count findings by severity."""
    from src.schemas.quality import IssueCounts

    return IssueCounts(
        info=sum(issue.severity is QualitySeverity.INFO for issue in issues),
        warning=sum(issue.severity is QualitySeverity.WARNING for issue in issues),
        error=sum(issue.severity is QualitySeverity.ERROR for issue in issues),
        critical=sum(issue.severity is QualitySeverity.CRITICAL for issue in issues),
    )


def calculate_quality_score(issues: tuple[QualityIssue, ...]) -> int:
    """Apply the documented transparent deduction table."""
    deduction = sum(SCORE_DEDUCTIONS[issue.severity] for issue in issues)
    return max(0, 100 - deduction)


def format_value(value: object) -> str:
    """Render a value safely for human-readable issue text."""
    if value is None:
        return "not provided"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value) or "empty"
    if isinstance(value, dict):
        return ", ".join(f"{key}={format_value(item)}" for key, item in value.items()) or "empty"
    return str(value)


def amounts_close(left: Decimal | None, right: Decimal | None, tolerance: Decimal) -> bool:
    """Compare two optional Decimal amounts using an inclusive tolerance."""
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance
