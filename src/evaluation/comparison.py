"""Ground-truth comparison with documented normalization rules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterator, Mapping

from src.schemas.common import FieldConfidenceEntry

from .schemas import ComparisonStatus, FieldComparison, GroundTruth

_METADATA_FIELDS = frozenset(
    {
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
)
_UNCERTAIN_MARKERS = ("?", "illegible", "unreadable", "unclear", "maybe", "possibly")
_NUMERIC_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")


def jsonable(value: Any) -> Any:
    """Return a JSON-safe representation without changing numeric text."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if not prefix and key_text in _METADATA_FIELDS:
                continue
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield from _flatten(child, path)
    elif isinstance(value, (list, tuple)):
        if not value:
            return
        for index, child in enumerate(value):
            yield from _flatten(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def flatten_payload(payload: Any) -> dict[str, Any]:
    """Flatten a typed extraction payload into ground-truth field paths."""
    if payload is None:
        return {}
    dumped = payload.model_dump(mode="python") if hasattr(payload, "model_dump") else payload
    return dict(_flatten(dumped))


def normalize_string(value: Any) -> str:
    """NFKC, trim, collapse whitespace, and casefold; punctuation is preserved."""
    text_value = value.value if isinstance(value, Enum) else value
    text = unicodedata.normalize("NFKC", str(text_value))
    return " ".join(text.split()).casefold()


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    text = str(value).strip().replace(",", "")
    if not _NUMERIC_RE.fullmatch(text):
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _date_parts(value: Any) -> tuple[date | None, datetime | None]:
    if isinstance(value, datetime):
        return value.date(), value
    if isinstance(value, date):
        return value, None
    text = str(value).strip()
    if not text:
        return None, None
    try:
        if "T" in text or " " in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.date(), parsed.replace(tzinfo=None)
        return date.fromisoformat(text), None
    except ValueError:
        return None, None


@dataclass(frozen=True)
class ComparisonOutcome:
    status: ComparisonStatus
    exact_match: bool
    normalized_match: bool
    method: str
    note: str = ""


def compare_values(
    expected: Any,
    extracted: Any,
    *,
    numeric_tolerance: Decimal = Decimal("0.01"),
) -> ComparisonOutcome:
    """Compare one expected value with one extracted value."""
    expected_decimal = _decimal(expected)
    extracted_decimal = _decimal(extracted)
    if expected_decimal is not None and extracted_decimal is not None:
        exact = expected_decimal == extracted_decimal
        match = abs(expected_decimal - extracted_decimal) <= numeric_tolerance
        return ComparisonOutcome(
            ComparisonStatus.CORRECT if match else ComparisonStatus.INCORRECT,
            exact,
            match,
            "numeric",
            f"absolute tolerance {numeric_tolerance}",
        )

    expected_date, expected_datetime = _date_parts(expected)
    extracted_date, extracted_datetime = _date_parts(extracted)
    if expected_date is not None and extracted_date is not None:
        if expected_datetime is not None and extracted_datetime is not None:
            exact = expected_datetime == extracted_datetime
        else:
            exact = expected_date == extracted_date
        return ComparisonOutcome(
            ComparisonStatus.CORRECT if exact else ComparisonStatus.INCORRECT,
            exact,
            exact,
            "date",
        )

    exact = jsonable(expected) == jsonable(extracted)
    match = normalize_string(expected) == normalize_string(extracted)
    return ComparisonOutcome(
        ComparisonStatus.CORRECT if match else ComparisonStatus.INCORRECT,
        exact,
        match,
        "string",
    )


def _is_uncertain(extracted: Any) -> bool:
    text = str(extracted).casefold()
    return any(marker in text for marker in _UNCERTAIN_MARKERS)


def _confidence_flagged(
    field_path: str,
    confidence: Mapping[str, FieldConfidenceEntry],
    threshold: float,
) -> tuple[bool, float | None]:
    entry = confidence.get(field_path)
    if entry is None:
        return False, None
    value = entry.confidence
    flagged = bool(entry.needs_review or (value is not None and value < threshold))
    return flagged, value


def evaluate_fields(
    truth: GroundTruth,
    result: Any,
    *,
    confidence_threshold: float,
    numeric_tolerance: Decimal = Decimal("0.01"),
) -> list[FieldComparison]:
    """Evaluate all ground-truth fields for one extraction result."""
    extracted_fields = flatten_payload(getattr(result, "data", None))
    confidence = getattr(result, "field_confidence", {}) or {}
    comparisons: list[FieldComparison] = []
    for field_path, expected in sorted(truth.expected_fields.items()):
        present = field_path in extracted_fields
        extracted = extracted_fields.get(field_path)
        flagged, confidence_value = _confidence_flagged(field_path, confidence, confidence_threshold)
        uncertain = flagged or _is_uncertain(extracted)

        if expected is None and (not present or extracted is None):
            outcome = ComparisonOutcome(ComparisonStatus.CORRECT, True, True, "null")
        elif expected is None:
            outcome = ComparisonOutcome(ComparisonStatus.EXTRA, False, False, "null")
        elif not present or extracted is None:
            outcome = ComparisonOutcome(ComparisonStatus.MISSING, False, False, "presence")
        else:
            outcome = compare_values(expected, extracted, numeric_tolerance=numeric_tolerance)

        status = outcome.status
        if uncertain and status in (ComparisonStatus.INCORRECT, ComparisonStatus.MISSING):
            status = ComparisonStatus.UNCERTAIN

        comparisons.append(
            FieldComparison(
                field_path=field_path,
                expected_value=jsonable(expected),
                extracted_value=jsonable(extracted) if present else None,
                status=status,
                exact_match=outcome.exact_match,
                normalized_match=outcome.normalized_match,
                comparison_method=outcome.method,
                uncertain=uncertain,
                confidence=confidence_value,
                note=outcome.note,
            )
        )

    for field_path, extracted in sorted(extracted_fields.items()):
        if field_path in truth.expected_fields or extracted is None:
            continue
        flagged, confidence_value = _confidence_flagged(field_path, confidence, confidence_threshold)
        comparisons.append(
            FieldComparison(
                field_path=field_path,
                expected_value=None,
                extracted_value=jsonable(extracted),
                status=ComparisonStatus.EXTRA,
                exact_match=False,
                normalized_match=False,
                comparison_method="presence",
                uncertain=flagged,
                confidence=confidence_value,
                note="field present in extraction but absent from ground truth",
            )
        )
    return comparisons
