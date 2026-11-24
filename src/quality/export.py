"""Safe exports for quality reports and reviewed output."""

from __future__ import annotations

import json
from typing import Any

from src.extraction.export import dump_extraction_result
from src.schemas.extraction_result import ExtractionResult
from src.schemas.quality import (
    QualityReport,
    ReviewedOutput,
    ReviewState,
)

from .review import apply_review_to_report

QUALITY_SCHEMA_VERSION = "4.0"


def _dumps(payload: Any, indent: int) -> str:
    return json.dumps(
        payload,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
    )


def quality_report_to_dict(
    report: QualityReport,
    state: ReviewState | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe quality report with optional session review state."""
    payload: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "quality_assessment": apply_review_to_report(report, state).model_dump(mode="json"),
    }
    if state is not None:
        payload["review_state"] = state.model_dump(mode="json")
    return payload


def dump_quality_report(
    report: QualityReport,
    state: ReviewState | None = None,
    *,
    indent: int = 2,
) -> str:
    return _dumps(quality_report_to_dict(report, state), indent)


def reviewed_output_to_dict(output: ReviewedOutput) -> dict[str, Any]:
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "reviewed_output": output.model_dump(mode="json"),
    }


def dump_reviewed_output(output: ReviewedOutput, *, indent: int = 2) -> str:
    return _dumps(reviewed_output_to_dict(output), indent)


def dump_original_extraction(result: ExtractionResult, *, indent: int = 2) -> str:
    return dump_extraction_result(result, indent=indent)


def _cell(value: Any, limit: int = 300) -> str:
    if value is None:
        text = "not provided"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_cell(item, limit) for item in value) or "empty"
    elif isinstance(value, dict):
        text = ", ".join(f"{key}={_cell(item, limit)}" for key, item in value.items()) or "empty"
    else:
        text = str(value)
    text = text.replace("\r", " ").replace("\n", "<br>").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_quality_markdown(
    report: QualityReport,
    result: ExtractionResult,
    state: ReviewState | None = None,
    output: ReviewedOutput | None = None,
) -> str:
    """Render a deterministic Markdown quality report without secrets or bytes."""
    reviewed_report = apply_review_to_report(report, state)
    lines = [
        "# Extraction quality report",
        "",
        f"- Document ID: {_cell(report.document_id)}",
        f"- Result ID: {_cell(report.result_id)}",
        f"- Document type: {_cell(report.document_type.value)}",
        f"- Quality status: **{_cell(report.quality_status.value)}**",
        f"- Review required: **{'yes' if report.review_required else 'no'}**",
        f"- Rule-based quality indicator: {report.quality_score}/100",
        f"- Assessment timestamp: {_cell(report.assessment_timestamp.isoformat())}",
        "",
        "> Quality status and score are rule-based indicators, not AI accuracy, "
        "correctness probability, or calibrated confidence.",
        "",
        "## Assessment summary",
        "",
        _cell(reviewed_report.assessment_summary, limit=3000).replace("<br>", " "),
        "",
        "Confidence values in this report are model estimates, not calibrated probabilities.",
        "",
        "## Severity counts",
        "",
        "| Severity | Count |",
        "| --- | ---: |",
        f"| Critical | {report.issue_counts.critical} |",
        f"| Error | {report.issue_counts.error} |",
        f"| Warning | {report.issue_counts.warning} |",
        f"| Info | {report.issue_counts.info} |",
        "",
        "## Issues",
        "",
    ]
    if reviewed_report.issues:
        lines.extend(
            [
                "| Issue ID | Rule | Severity | Category | Description | Fields | Expected | Actual | Action | Resolved |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for issue in reviewed_report.issues:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(issue.issue_id),
                        _cell(issue.rule_id),
                        _cell(issue.severity.value),
                        _cell(issue.category.value),
                        _cell(issue.description),
                        _cell(issue.affected_fields),
                        _cell(issue.expected_value),
                        _cell(issue.actual_value),
                        _cell(issue.recommended_action),
                        "yes" if issue.resolved else "no",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No quality issues were detected.")

    lines.extend(["", "## Confidence and field review", ""])
    if reviewed_report.field_review:
        lines.extend(
            [
                "| Field | Value | Confidence estimate | Evidence | Severity | Reason |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for field in reviewed_report.field_review:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(field.field_path),
                        _cell(field.value),
                        _cell(field.confidence),
                        _cell(field.evidence),
                        _cell(field.severity.value),
                        _cell(field.reason),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No confidence-related fields required review.")

    lines.extend(["", "## Processing warnings", ""])
    warnings = list(reviewed_report.validation_summary.warnings)
    if warnings:
        lines.extend(f"- {_cell(warning, limit=1000)}" for warning in warnings)
    else:
        lines.append("- None")

    lines.extend(["", "## Review changes", ""])
    if state and (state.field_edits or state.issue_reviews or state.reviewer_notes):
        lines.append("Review changes are session-only and do not modify the original extraction.")
        if state.field_edits:
            lines.extend(["", "| Field | Original AI value | Reviewed value | Note |", "| --- | --- | --- | --- |"])
            for path, edit in sorted(state.field_edits.items()):
                lines.append(
                    f"| {_cell(path)} | {_cell(edit.original_value)} | "
                    f"{_cell(edit.edited_value)} | {_cell(edit.note)} |"
                )
        if state.issue_reviews:
            lines.extend(["", "| Issue | Review status | Note |", "| --- | --- | --- |"])
            for issue_id, review in sorted(state.issue_reviews.items()):
                lines.append(
                    f"| {_cell(issue_id)} | {_cell(review.status.value)} | {_cell(review.note)} |"
                )
        if state.reviewer_notes:
            lines.extend(
                [""] + ["- Reviewer note: " + _cell(note, limit=1000) for note in state.reviewer_notes]
            )
    else:
        lines.append("No review changes were recorded.")

    if output is not None:
        lines.extend(
            [
                "",
                "## Reviewed output",
                "",
                f"- Changed fields: {_cell(output.changed_fields)}",
                f"- Verified against source: **{'yes' if output.verified_against_source else 'no'}**",
                "- " + output.verification_note,
            ]
        )

    lines.extend(
        [
            "",
            "## Original extraction",
            "",
            "The original AI extraction remains immutable during review.",
            "",
            f"- Original JSON export available separately for result `{result.result_id}`.",
            "",
            f"Generated from rule version {report.rule_version}.",
        ]
    )
    return "\n".join(lines) + "\n"
