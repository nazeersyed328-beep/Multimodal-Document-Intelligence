"""Schemas for model-facing extraction requests and parsed model output.

These models describe the untrusted response envelope returned by a vision
model. The document-specific values inside ``data`` are validated separately
against the strict Phase 1 document schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import (
    ConfidenceScore,
    DocumentType,
    StrictModel,
)


class DocumentTypeDecision(StrictModel):
    """Structured document-type classification returned by the model."""

    document_type: DocumentType
    confidence: ConfidenceScore | None = None
    reason: str | None = Field(default=None, max_length=1000)


class ModelFieldConfidence(StrictModel):
    """One model-estimated confidence entry for a named field."""

    field_path: str = Field(min_length=1, max_length=500)
    confidence: ConfidenceScore | None = None
    evidence: str | None = Field(default=None, max_length=2000)
    needs_review: bool = False


class ExtractionEnvelope(StrictModel):
    """Untrusted extraction envelope parsed from a model response."""

    document_type: DocumentType
    data: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    unreadable_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    overall_confidence: ConfidenceScore | None = None
    confidence_note: str | None = Field(default=None, max_length=1000)
    field_confidence: list[ModelFieldConfidence] = Field(default_factory=list)
