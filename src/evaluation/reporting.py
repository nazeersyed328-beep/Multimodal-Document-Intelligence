"""Markdown, JSON, and CSV report generation for Phase 5 evaluations."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import EvaluationRunSummary

_REPORT_SCHEMA_VERSION = "5.0"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:160]


def _display(value: Any) -> str:
    if value is None:
        return "not measurable"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _percent(value: Any) -> str:
    if value is None:
        return "not measurable"
    return f"{float(value) * 100:.2f}%"


def _binary_rows(section: Mapping[str, Any]) -> list[str]:
    rows = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| True positives | {section.get('true_positives', 0)} |",
        f"| False positives | {section.get('false_positives', 0)} |",
        f"| False negatives | {section.get('false_negatives', 0)} |",
        f"| Support | {section.get('support', 0)} |",
        f"| Precision | {_percent(section.get('precision'))} |",
        f"| Recall | {_percent(section.get('recall'))} |",
        f"| F1 | {_display(section.get('f1'))} |",
        f"| Measurable | {section.get('measurable', False)} |",
    ]
    return rows


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    header = list(headers)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [_display(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_markdown_report(summary: EvaluationRunSummary) -> str:
    schema = summary.schema_validity
    fields = summary.field_metrics
    doc_types = summary.document_type_metrics
    quality = summary.quality_metrics
    perf = summary.performance
    validation = summary.dataset_validation
    lines: list[str] = [
        "# Phase 5 evaluation report",
        "",
        "> This report separates local synthetic tests, mocked API tests, and real Gemini API evaluation. "
        "It does not present rule-based quality scores as model accuracy.",
        "",
        "## 1. Report metadata",
        "",
        f"- Report schema version: `{_REPORT_SCHEMA_VERSION}`",
        f"- Run ID: `{summary.run_id}`",
        f"- Generated at: `{summary.generated_at.isoformat()}`",
        f"- Evaluation mode: **{summary.mode.value}**",
        f"- API status: `{summary.api_status}`",
        f"- Model ID: `{summary.model_id}`",
        f"- Dataset version: `{summary.dataset_version}`",
        f"- Subset: `{summary.subset.value}`",
        f"- Raw results directory: `{summary.raw_results_directory}`",
        f"- Resumed documents: {summary.resumed_count}",
        "",
        "## 2. Dataset and provenance",
        "",
        f"- Dataset version: `{validation.get('dataset_version', summary.dataset_version)}`",
        f"- Sample count: {validation.get('sample_count', summary.sample_count)}",
        f"- Document type counts: {_display(validation.get('document_type_counts', {}))}",
        f"- Subset counts: {_display(validation.get('subset_counts', {}))}",
        f"- Validation errors: {len(validation.get('errors', []))}",
        f"- Validation warnings: {len(validation.get('warnings', []))}",
        "- Source and license: synthetic artwork generated locally; manually authored ground truth; CC0-1.0 for these synthetic assets.",
        "",
    ]
    if validation.get("errors"):
        lines.extend(["Validation errors:", ""] + [f"- {item}" for item in validation["errors"]] + [""])
    if validation.get("warnings"):
        lines.extend(["Validation warnings:", ""] + [f"- {item}" for item in validation["warnings"]] + [""])

    lines.extend(
        [
            "## 3. Execution mode and API disclosure",
            "",
            f"- Mode: **{summary.mode.value}**",
            f"- API calls: {'live Gemini API calls were used' if summary.mode.value == 'real_gemini_api_evaluation' else 'no live Gemini API calls were made'}",
            f"- Model and timestamp are recorded above so this run can be reproduced or compared.",
            "",
            "## 4. Schema validity",
            "",
        ]
    )
    lines.extend(
        _table(
            ("Valid", "Invalid", "Total", "Schema validity rate"),
            (
                (
                    schema.get("valid_count", 0),
                    schema.get("invalid_count", 0),
                    schema.get("total", 0),
                    _percent(schema.get("schema_validity_rate")),
                ),
            ),
        )
    )
    lines.extend(
        [
            "",
            "Schema validity means the provider output matched the strict typed document schema; it does not prove extracted values are correct.",
            "",
            "## 5. Field metrics",
            "",
            f"- Compared fields: {fields.get('compared_fields', 0)}",
            f"- Exact-match accuracy: {_percent(fields.get('exact_match_accuracy'))}",
            f"- Normalized accuracy: {_percent(fields.get('normalized_accuracy'))}",
            f"- Missing-field rate: {_percent(fields.get('missing_field_rate'))}",
            f"- Incorrectly populated field rate: {_percent(fields.get('incorrectly_populated_field_rate'))}",
            "",
        ]
    )
    lines.extend(
        _table(
            ("Status", "Count"),
            ((key, value) for key, value in sorted(fields.get("status_counts", {}).items())),
        )
    )
    lines.extend(
        [
            "",
            "Normalization: casefolded NFKC text with trimmed and collapsed whitespace; numeric values compared with the configured absolute tolerance; dates compared as ISO calendar dates unless both values include time.",
            "",
            "## 6. Document type accuracy",
            "",
        ]
    )
    lines.extend(
        _table(
            ("Correct", "Incorrect", "Total", "Accuracy"),
            (
                (
                    doc_types.get("correct", 0),
                    doc_types.get("incorrect", 0),
                    doc_types.get("total", 0),
                    _percent(doc_types.get("document_type_accuracy")),
                ),
            ),
        )
    )
    lines.extend(["", "## 7. Per-document-type metrics", ""])
    for document_type, section in sorted(summary.per_document_type_metrics.items()):
        type_fields = section.get("field_metrics", {})
        lines.extend(
            [
                f"### {document_type}",
                "",
                f"- Samples: {section.get('sample_count', 0)}",
                f"- Schema validity: {_percent(section.get('schema_validity', {}).get('schema_validity_rate'))}",
                f"- Exact-match accuracy: {_percent(type_fields.get('exact_match_accuracy'))}",
                f"- Normalized accuracy: {_percent(type_fields.get('normalized_accuracy'))}",
                f"- Average latency: {_display(section.get('performance', {}).get('average_latency_seconds'))} s",
                "",
            ]
        )
    lines.extend(
        [
            "## 8. Quality-system metrics",
            "",
            f"- Detected issue rules: {quality.get('detected_issue_count', 0)}",
            f"- Expected issue rules: {quality.get('expected_issue_count', 0)}",
            "",
            "### Issue rules (micro)",
            "",
        ]
    )
    lines.extend(_binary_rows(quality.get("issue_rules_micro", {})))
    for label in ("review_required", "low_confidence_fields", "unreadable_fields"):
        lines.extend(["", f"### {label.replace('_', ' ')}", ""])
        lines.extend(_binary_rows(quality.get(label, {})))
    lines.extend(["", quality.get("notes", ""), "", "## 9. Performance", ""])
    lines.extend(
        _table(
            ("Metric", "Value"),
            (
                ("Documents", perf.get("documents", 0)),
                ("API request success rate", _percent(perf.get("api_request_success_rate"))),
                ("API request failure rate", _percent(perf.get("api_request_failure_rate"))),
                ("Total processing seconds", _display(perf.get("total_processing_seconds"))),
                ("Average latency seconds", _display(perf.get("average_latency_seconds"))),
                ("Max latency seconds", _display(perf.get("max_latency_seconds"))),
                ("Provider requests", perf.get("provider_requests", 0)),
                ("Estimated retries", perf.get("retry_count", 0)),
            ),
        )
    )
    lines.extend(
        [
            "",
            "Latency includes document loading, preprocessing, all provider requests, parsing, and local quality assessment.",
            "",
            "## 10. Error analysis",
            "",
            f"- Error entries: {len(summary.error_analysis)}",
            "",
        ]
    )
    if summary.error_analysis:
        rows = (
            (
                error.sample_id,
                error.error_category,
                error.field_path,
                error.flagged_for_review,
                error.potential_cause,
            )
            for error in summary.error_analysis[:200]
        )
        lines.extend(_table(("Sample", "Category", "Field", "Flagged", "Potential cause"), rows))
        if len(summary.error_analysis) > 200:
            lines.extend(["", f"Only the first 200 of {len(summary.error_analysis)} errors are shown."])
    else:
        lines.append("No comparison or quality-rule errors were recorded.")
    lines.extend(["", "## 11. Sample-level outcomes", "",])
    lines.extend(
        _table(
            ("Sample", "Type", "Schema", "Fields correct", "Issues detected", "Review", "Error"),
            (
                (
                    item.sample_id,
                    item.document_type,
                    "yes" if item.schema_valid else "no",
                    f"{item.field_metrics.get('correct', 0)}/{item.field_metrics.get('compared_fields', 0)}",
                    ", ".join(item.detected_issue_rules) or "none",
                    _display(item.quality_review_required),
                    item.error_code or "none",
                )
                for item in summary.document_evaluations
            ),
        )
    )
    lines.extend(["", "## 12. Limitations", ""])
    lines.extend([f"- {item}" for item in summary.limitations] or ["- None recorded."])
    lines.extend(["", "## 13. Recommendations", ""])
    lines.extend([f"- {item}" for item in summary.recommendations] or ["- None recorded."])
    lines.extend(
        [
            "",
            "## 14. Reproducibility notes",
            "",
            "- Dataset sample hashes and ground truth are stored under `evaluation/`.",
            "- Raw extraction and quality outputs are stored per sample in `evaluation/results/`.",
            "- Summary JSON uses the same run ID as the results directory.",
            "- Re-running with the same dataset, settings, model, and mode reuses matching raw outputs when resume is enabled.",
            "- Live API mode requires explicit opt-in and is never triggered by opening the application.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(summary: EvaluationRunSummary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(summary), encoding="utf-8")
    return path


def write_summary_json(summary: EvaluationRunSummary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "evaluation": summary.model_dump(mode="json"),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_error_csv(summary: EvaluationRunSummary, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "sample_id",
                "document_type",
                "field_path",
                "expected_value",
                "actual_value",
                "error_category",
                "flagged_for_review",
                "potential_cause",
            )
        )
        for error in summary.error_analysis:
            writer.writerow(
                (
                    error.sample_id,
                    error.document_type,
                    error.field_path,
                    _display(error.expected_value),
                    _display(error.actual_value),
                    error.error_category,
                    _display(error.flagged_for_review),
                    error.potential_cause,
                )
            )
    return path


def write_reports(
    summary: EvaluationRunSummary,
    reports_root: Path,
    *,
    formats: Iterable[str] = ("json", "markdown", "csv"),
) -> dict[str, str]:
    requested = {item.lower() for item in formats}
    unknown = requested - {"json", "markdown", "csv"}
    if unknown:
        raise ValueError(f"Unsupported report format(s): {', '.join(sorted(unknown))}")
    prefix = _safe_name(f"{summary.run_id}_{summary.mode.value}")
    outputs: dict[str, str] = {}
    if "json" in requested:
        outputs["json"] = str(write_summary_json(summary, reports_root / f"{prefix}_summary.json"))
    if "markdown" in requested:
        outputs["markdown"] = str(write_markdown_report(summary, reports_root / f"{prefix}_report.md"))
    if "csv" in requested:
        outputs["csv"] = str(write_error_csv(summary, reports_root / f"{prefix}_errors.csv"))
    return outputs
