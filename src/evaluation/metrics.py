"""Deterministic metric calculations for Phase 5 evaluation results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .schemas import ComparisonStatus, DocumentEvaluation, ErrorEntry, GroundTruth


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _binary_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    support = tp + fn
    predicted = tp + fp
    measurable = support > 0 or predicted > 0
    precision = tp / predicted if predicted else None
    recall = tp / support if support else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "support": support,
        "measurable": measurable,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_schema_validity(evaluations: Iterable[DocumentEvaluation]) -> dict[str, Any]:
    items = list(evaluations)
    valid = sum(item.schema_valid for item in items)
    invalid = len(items) - valid
    return {
        "valid_count": valid,
        "invalid_count": invalid,
        "total": len(items),
        "schema_validity_rate": _safe_rate(valid, len(items)),
    }


def compute_field_metrics_from_comparisons(comparisons: Iterable[Any]) -> dict[str, Any]:
    counts = {status.value: 0 for status in ComparisonStatus}
    exact = 0
    normalized = 0
    total = 0
    fields_with_non_null_expectation = 0
    uncertain_count = 0
    for comparison in comparisons:
        total += 1
        counts[comparison.status.value] += 1
        exact += int(comparison.exact_match)
        normalized += int(comparison.normalized_match)
        uncertain_count += int(comparison.uncertain)
        if comparison.expected_value is not None:
            fields_with_non_null_expectation += 1
    incorrect_populated = counts["incorrect"] + counts["extra"]
    return {
        "compared_fields": total,
        "exact_matches": exact,
        "exact_match_accuracy": _safe_rate(exact, total),
        "normalized_matches": normalized,
        "normalized_accuracy": _safe_rate(normalized, total),
        "status_counts": counts,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "missing": counts["missing"],
        "extra": counts["extra"],
        "uncertain": uncertain_count,
        "missing_field_rate": _safe_rate(counts["missing"], fields_with_non_null_expectation),
        "incorrectly_populated_field_rate": _safe_rate(incorrect_populated, total),
        "fields_with_non_null_expectation": fields_with_non_null_expectation,
    }


def compute_field_metrics(evaluations: Iterable[DocumentEvaluation]) -> dict[str, Any]:
    return compute_field_metrics_from_comparisons(
        comparison
        for item in evaluations
        for comparison in item.comparisons
    )


def compute_document_type_metrics(evaluations: Iterable[DocumentEvaluation]) -> dict[str, Any]:
    items = list(evaluations)
    correct = sum(item.document_type == item.expected_document_type for item in items)
    return {
        "correct": correct,
        "incorrect": len(items) - correct,
        "total": len(items),
        "document_type_accuracy": _safe_rate(correct, len(items)),
    }


def compute_performance(evaluations: Iterable[DocumentEvaluation]) -> dict[str, Any]:
    items = list(evaluations)
    successes = sum(item.error_code is None for item in items)
    failures = len(items) - successes
    latencies = [item.latency_seconds for item in items]
    return {
        "documents": len(items),
        "api_request_success_count": successes,
        "api_request_failure_count": failures,
        "api_request_success_rate": _safe_rate(successes, len(items)),
        "api_request_failure_rate": _safe_rate(failures, len(items)),
        "total_processing_seconds": round(sum(latencies), 6),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "max_latency_seconds": round(max(latencies), 6) if latencies else None,
        "provider_requests": sum(item.provider_requests for item in items),
        "retry_count": sum(item.retry_count for item in items),
        "average_provider_requests": (
            sum(item.provider_requests for item in items) / len(items) if items else None
        ),
    }


def compute_per_document_type_metrics(evaluations: Iterable[DocumentEvaluation]) -> dict[str, Any]:
    grouped: dict[str, list[DocumentEvaluation]] = {}
    for item in evaluations:
        grouped.setdefault(item.expected_document_type, []).append(item)
    return {
        document_type: {
            "sample_count": len(items),
            "schema_validity": compute_schema_validity(items),
            "field_metrics": compute_field_metrics(items),
            "document_type": compute_document_type_metrics(items),
            "performance": compute_performance(items),
        }
        for document_type, items in sorted(grouped.items())
    }


def compute_quality_metrics(
    evaluations: Iterable[DocumentEvaluation],
    truths: Mapping[str, GroundTruth],
) -> dict[str, Any]:
    items = list(evaluations)
    rule_tp = rule_fp = rule_fn = 0
    per_rule: dict[str, dict[str, Any]] = {}
    all_rules = sorted(
        {rule for item in items for rule in set(item.detected_issue_rules) | set(item.expected_issue_rules)}
    )
    for rule in all_rules:
        tp = sum(rule in item.detected_issue_rules and rule in item.expected_issue_rules for item in items)
        fp = sum(rule in item.detected_issue_rules and rule not in item.expected_issue_rules for item in items)
        fn = sum(rule not in item.detected_issue_rules and rule in item.expected_issue_rules for item in items)
        per_rule[rule] = _binary_metrics(tp, fp, fn)
        rule_tp += tp
        rule_fp += fp
        rule_fn += fn

    review_pairs = [
        item
        for item in items
        if item.expected_review_required is not None and item.quality_review_required is not None
    ]
    review_tp = sum(
        item.expected_review_required and item.quality_review_required for item in review_pairs
    )
    review_fp = sum(
        not item.expected_review_required and item.quality_review_required for item in review_pairs
    )
    review_fn = sum(
        item.expected_review_required and not item.quality_review_required for item in review_pairs
    )

    low_conf_tp = low_conf_fp = low_conf_fn = 0
    unreadable_tp = unreadable_fp = unreadable_fn = 0
    for item in items:
        truth = truths.get(item.sample_id)
        if truth is None:
            continue
        expected_low = set(truth.expected_low_confidence_fields)
        detected_low = set(item.flagged_low_confidence_fields)
        low_conf_tp += len(expected_low & detected_low)
        low_conf_fp += len(detected_low - expected_low)
        low_conf_fn += len(expected_low - detected_low)

        expected_unreadable = set(truth.expected_unreadable_fields)
        detected_unreadable = set(item.flagged_unreadable_fields)
        unreadable_tp += len(expected_unreadable & detected_unreadable)
        unreadable_fp += len(detected_unreadable - expected_unreadable)
        unreadable_fn += len(expected_unreadable - detected_unreadable)

    return {
        "detected_issue_count": sum(len(item.detected_issue_rules) for item in items),
        "expected_issue_count": sum(len(item.expected_issue_rules) for item in items),
        "issue_rules_micro": _binary_metrics(rule_tp, rule_fp, rule_fn),
        "per_rule": per_rule,
        "review_required": _binary_metrics(review_tp, review_fp, review_fn),
        "low_confidence_fields": _binary_metrics(low_conf_tp, low_conf_fp, low_conf_fn),
        "unreadable_fields": _binary_metrics(unreadable_tp, unreadable_fp, unreadable_fn),
        "notes": (
            "Issue-rule metrics compare expected rule IDs with deterministic Phase 4 findings. "
            "Low-confidence and unreadable-field metrics compare expected field paths with model flags; "
            "small supports limit statistical meaning."
        ),
    }


def _cause_for_comparison(comparison: Any) -> str:
    if comparison.comparison_method == "numeric":
        return "Review printed numbers and line-item arithmetic."
    if comparison.comparison_method == "date":
        return "Review date format, timezone handling, and source legibility."
    if comparison.status is ComparisonStatus.MISSING:
        return "Field may be absent, obscured, or skipped by the model."
    if comparison.status is ComparisonStatus.EXTRA:
        return "Model may have inferred a value not present in the source."
    return "Review source rendering and extraction prompt adherence."


def build_error_analysis(
    evaluations: Iterable[DocumentEvaluation],
    truths: Mapping[str, GroundTruth],
) -> list[ErrorEntry]:
    items = list(evaluations)
    errors: list[ErrorEntry] = []
    for item in items:
        flagged = bool(item.quality_review_required)
        truth = truths.get(item.sample_id)
        for comparison in item.comparisons:
            if comparison.status is ComparisonStatus.CORRECT:
                continue
            errors.append(
                ErrorEntry(
                    sample_id=item.sample_id,
                    document_type=item.document_type,
                    field_path=comparison.field_path,
                    expected_value=comparison.expected_value,
                    actual_value=comparison.extracted_value,
                    error_category=comparison.status.value,
                    flagged_for_review=flagged,
                    potential_cause=_cause_for_comparison(comparison),
                )
            )
        if item.document_type != item.expected_document_type:
            errors.append(
                ErrorEntry(
                    sample_id=item.sample_id,
                    document_type=item.document_type,
                    field_path="document_type",
                    expected_value=item.expected_document_type,
                    actual_value=item.document_type,
                    error_category="document_classification_error",
                    flagged_for_review=flagged,
                    potential_cause="Review auto-detection prompt and ambiguous visual cues.",
                )
            )
        if not item.schema_valid or item.error_code:
            errors.append(
                ErrorEntry(
                    sample_id=item.sample_id,
                    document_type=item.document_type,
                    field_path="extraction_result",
                    expected_value="valid typed extraction",
                    actual_value=item.error_code or item.extraction_status,
                    error_category="schema_or_provider_failure",
                    flagged_for_review=flagged,
                    potential_cause="Review provider response constraints and safe parser errors.",
                )
            )
        if truth is not None:
            for rule in set(item.detected_issue_rules) - set(truth.expected_issue_rules):
                errors.append(
                    ErrorEntry(
                        sample_id=item.sample_id,
                        document_type=item.document_type,
                        field_path="quality_issue",
                        expected_value=None,
                        actual_value=rule,
                        error_category="false_positive_quality_warning",
                        flagged_for_review=flagged,
                        potential_cause="Confirm whether the rule fired on a model-extraction artifact.",
                    )
                )
            for rule in set(truth.expected_issue_rules) - set(item.detected_issue_rules):
                errors.append(
                    ErrorEntry(
                        sample_id=item.sample_id,
                        document_type=item.document_type,
                        field_path="quality_issue",
                        expected_value=rule,
                        actual_value=None,
                        error_category="false_negative_quality_warning",
                        flagged_for_review=flagged,
                        potential_cause="Confirm whether extraction hid the source inconsistency.",
                    )
                )
    return errors
