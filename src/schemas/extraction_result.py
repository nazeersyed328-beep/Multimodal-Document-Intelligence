"""Unified extraction result schema.

The payload is a discriminated union of the four concrete document schemas, so
document-specific data is always typed and never an arbitrary dictionary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import Field, model_validator

from .common import (
    ConfidenceScore,
    DocumentId,
    DocumentType,
    ErrorDetail,
    FieldConfidenceEntry,
    StrictModel,
    ValidationStatus,
)
from .diagram import DiagramDocument
from .handwritten_note import HandwrittenNoteDocument
from .invoice import InvoiceDocument
from .receipt import ReceiptDocument


class ProcessingStatus(str, Enum):
    """Lifecycle state of an extraction attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


DocumentPayload = Annotated[
    InvoiceDocument | ReceiptDocument | HandwrittenNoteDocument | DiagramDocument,
    Field(discriminator="document_type"),
]


class ExtractionResult(StrictModel):
    """One validation/extraction outcome with confidence and review metadata."""

    result_id: DocumentId
    document_type: DocumentType
    source_document_id: DocumentId | None = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    data: DocumentPayload | None = None
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    validation_errors: list[ErrorDetail] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    field_confidence: dict[str, FieldConfidenceEntry] = Field(default_factory=dict)
    overall_confidence: ConfidenceScore | None = None
    needs_review: bool = False
    processed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_payload_type(self) -> ExtractionResult:
        if self.data is not None and self.data.document_type.value != self.document_type.value:
            raise ValueError("data.document_type must match document_type.")
        return self
