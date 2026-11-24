"""Reusable types, enums, and base models shared by every document schema.

Pydantic validates structure and types only. It does not verify that a value
was read correctly from a document, and it never invents a missing value.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Generic, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)

DECIMAL_PATTERN = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


class DocumentType(str, Enum):
    """Supported and explicitly unknown document categories."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    HANDWRITTEN_NOTE = "handwritten_note"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Outcome of structural validation for an extracted document."""

    NOT_VALIDATED = "not_validated"
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"


class ErrorSeverity(str, Enum):
    """Severity of a validation error or warning detail."""

    ERROR = "error"
    WARNING = "warning"


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _parse_decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric values.")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        raise ValueError(
            "Floating-point amounts are rejected to avoid precision loss; "
            "pass a Decimal, an integer, or a numeric string."
        )
    if isinstance(value, str):
        candidate = value.strip()
        if not DECIMAL_PATTERN.fullmatch(candidate):
            raise ValueError(
                f"{value!r} is not a valid decimal amount; "
                "use digits with an optional decimal point and optional thousands separators."
            )
        return Decimal(candidate.replace(",", ""))
    raise ValueError(
        f"Expected a Decimal, integer, or numeric string; got {type(value).__name__}."
    )


def _reject_bool(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric values.")
    return value


def _check_source_filename(value: str) -> str:
    if value in {".", ".."} or any(char in value for char in ("/", "\\", "\x00")):
        raise ValueError(
            "Source filename must be a bare filename without path separators."
        )
    return value


MonetaryAmount = Annotated[
    Decimal,
    BeforeValidator(_parse_decimal),
    Field(ge=0, description="Nonnegative monetary amount stored as Decimal."),
]

QuantityValue = Annotated[
    Decimal,
    BeforeValidator(_parse_decimal),
    Field(gt=0, description="Positive quantity stored as Decimal."),
]

ConfidenceScore = Annotated[
    float,
    BeforeValidator(_reject_bool),
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]

CurrencyCode = Annotated[
    str,
    Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217-style uppercase three-letter currency code.",
    ),
]

DocumentId = Annotated[str, Field(min_length=1, max_length=128)]

SourceFilename = Annotated[
    str,
    Field(min_length=1, max_length=255),
    AfterValidator(_check_source_filename),
]

ExtractedValue = str | int | Decimal | bool | date | datetime | list[str]

T = TypeVar("T")


class FieldConfidence(StrictModel, Generic[T]):
    """Optional confidence metadata for one extracted field."""

    value: T | None = None
    confidence: ConfidenceScore | None = None
    evidence: str | None = Field(default=None, max_length=2000)
    needs_review: bool = False


FieldConfidenceEntry = FieldConfidence[ExtractedValue]


class ErrorDetail(StrictModel):
    """A single structural validation problem with an actionable message."""

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    field_path: str | None = Field(default=None, max_length=500)
    severity: ErrorSeverity = ErrorSeverity.ERROR


class DocumentBase(StrictModel):
    """Metadata shared by every typed document payload."""

    document_id: DocumentId
    document_type: DocumentType
    source_filename: SourceFilename
    extracted_at: datetime
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorDetail] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    unreadable_fields: list[str] = Field(default_factory=list)
