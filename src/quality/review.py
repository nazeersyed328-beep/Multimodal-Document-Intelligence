"""Session-only human review state helpers.

Review state never mutates the original AI extraction. Edits are recorded
separately and can be reset or exported as a clearly labeled reviewed output.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.schemas.extraction_result import ExtractionResult
from src.schemas.quality import (
    FieldEdit,
    IssueReview,
    IssueReviewStatus,
    QualityReport,
    ReviewedOutput,
    ReviewState,
)

_PATH_PARTS = re.compile(r"[^.\[\]]+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    return value


def _empty_container(next_part: str) -> dict[str, Any] | list[Any]:
    return [] if next_part.isdigit() else {}


def _set_value(payload: dict[str, Any], field_path: str, value: Any) -> None:
    parts = _PATH_PARTS.findall(field_path)
    if not parts:
        return
    cursor: Any = payload
    for position, part in enumerate(parts[:-1]):
        next_part = parts[position + 1]
        if isinstance(cursor, dict):
            expected_list = next_part.isdigit()
            container = cursor.get(part)
            if (
                container is None
                or (expected_list and not isinstance(container, list))
                or (not expected_list and not isinstance(container, dict))
            ):
                container = _empty_container(next_part)
                cursor[part] = container
            cursor = container
        elif isinstance(cursor, list):
            if not part.isdigit():
                return
            index = int(part)
            while len(cursor) <= index:
                cursor.append(_empty_container(next_part))
            cursor = cursor[index]
        else:
            return
    last = parts[-1]
    if isinstance(cursor, dict):
        cursor[last] = value
    elif isinstance(cursor, list) and last.isdigit():
        index = int(last)
        while len(cursor) <= index:
            cursor.append(None)
        cursor[index] = value


def new_review_state(result_id: str) -> ReviewState:
    return ReviewState(result_id=result_id, updated_at=_utc_now())


def get_or_create_review(states: dict[str, ReviewState], result_id: str) -> ReviewState:
    state = states.get(result_id)
    if state is None:
        state = new_review_state(result_id)
        states[result_id] = state
    return state


def set_issue_review(
    state: ReviewState,
    issue_id: str,
    status: IssueReviewStatus,
    note: str | None = None,
) -> ReviewState:
    issue_reviews = dict(state.issue_reviews)
    issue_reviews[issue_id] = IssueReview(
        issue_id=issue_id,
        status=status,
        note=(note or None) and note[:2000],
        reviewed_at=_utc_now(),
    )
    return state.model_copy(update={"issue_reviews": issue_reviews, "updated_at": _utc_now()})


def remove_issue_review(state: ReviewState, issue_id: str) -> ReviewState:
    issue_reviews = dict(state.issue_reviews)
    issue_reviews.pop(issue_id, None)
    return state.model_copy(update={"issue_reviews": issue_reviews, "updated_at": _utc_now()})


def set_field_edit(
    state: ReviewState,
    field_path: str,
    edited_value: Any,
    note: str | None = None,
    *,
    original_value: Any = None,
) -> ReviewState:
    field_edits = dict(state.field_edits)
    field_edits[field_path] = FieldEdit(
        field_path=field_path,
        original_value=original_value,  # type: ignore[arg-type]
        edited_value=edited_value,  # type: ignore[arg-type]
        note=(note or None) and note[:2000],
        edited_at=_utc_now(),
    )
    return state.model_copy(update={"field_edits": field_edits, "updated_at": _utc_now()})


def remove_field_edit(state: ReviewState, field_path: str) -> ReviewState:
    field_edits = dict(state.field_edits)
    field_edits.pop(field_path, None)
    return state.model_copy(update={"field_edits": field_edits, "updated_at": _utc_now()})


def add_reviewer_note(state: ReviewState, note: str) -> ReviewState:
    cleaned = note.strip()[:2000]
    if not cleaned:
        return state
    notes = tuple(state.reviewer_notes) + (cleaned,)
    return state.model_copy(update={"reviewer_notes": notes, "updated_at": _utc_now()})


def reset_review(state: ReviewState) -> ReviewState:
    return new_review_state(state.result_id)


def has_review_changes(state: ReviewState | None) -> bool:
    return bool(
        state
        and (state.field_edits or state.issue_reviews or state.reviewer_notes)
    )


def apply_review_to_report(report: QualityReport, state: ReviewState | None) -> QualityReport:
    """Mark detected issues resolved without removing or hiding them."""
    if state is None:
        return report
    resolved = {
        issue_id
        for issue_id, review in state.issue_reviews.items()
        if review.status is IssueReviewStatus.RESOLVED
    }
    if not resolved:
        return report
    issues = tuple(
        issue.model_copy(update={"resolved": issue.issue_id in resolved})
        for issue in report.issues
    )
    return report.model_copy(update={"issues": issues})


def review_progress(report: QualityReport, state: ReviewState | None) -> dict[str, int]:
    detected = len(report.issues)
    reviewed = len(state.issue_reviews) if state else 0
    resolved = sum(
        review.status is IssueReviewStatus.RESOLVED
        for review in (state.issue_reviews.values() if state else ())
    )
    return {"detected": detected, "reviewed": reviewed, "resolved": resolved}


def build_reviewed_output(
    result: ExtractionResult,
    report: QualityReport,
    state: ReviewState,
) -> ReviewedOutput:
    """Overlay session edits on a copy of the original extraction data."""
    data = result.data.model_dump(mode="json") if result.data is not None else {}
    for field_path, edit in sorted(state.field_edits.items()):
        _set_value(data, field_path, edit.edited_value)
    edits = tuple(state.field_edits[path] for path in sorted(state.field_edits))
    return ReviewedOutput(
        report_id=report.report_id,
        result_id=result.result_id,
        document_id=(result.source_document_id or (result.data.document_id if result.data else "unknown")),
        document_type=result.document_type,
        data=data,
        changed_fields=tuple(sorted(state.field_edits)),
        field_edits=edits,
        reviewer_notes=tuple(state.reviewer_notes),
        generated_at=_utc_now(),
    )
