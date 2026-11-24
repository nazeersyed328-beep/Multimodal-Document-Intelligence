"""Strict schema for handwritten notes.

Illegible handwriting is represented explicitly with markers. The model is
never allowed to guess a reading and present it as transcribed text.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .common import DocumentBase, DocumentType, StrictModel


class UnreadableReason(str, Enum):
    """Why a span of handwriting could not be read."""

    ILLEGIBLE = "illegible"
    OBSCURED = "obscured"
    CUT_OFF = "cut_off"
    OTHER = "other"


class UnreadableMarker(StrictModel):
    """An explicit placeholder for text that could not be read."""

    marker_id: str = Field(min_length=1, max_length=50)
    context: str | None = Field(default=None, max_length=1000)
    reason: UnreadableReason = UnreadableReason.ILLEGIBLE


class DateMention(StrictModel):
    """A date as written plus an optional confidently resolved value."""

    raw_text: str = Field(min_length=1, max_length=200)
    resolved_date: date | None = None


class ActionItem(StrictModel):
    """A task identified in a note."""

    text: str = Field(min_length=1, max_length=1000)
    assignee: str | None = Field(default=None, max_length=300)
    due_date: date | None = None


class HandwrittenNoteDocument(DocumentBase):
    """Typed payload for a handwritten note."""

    document_type: Literal[DocumentType.HANDWRITTEN_NOTE] = (
        DocumentType.HANDWRITTEN_NOTE
    )
    title: str | None = Field(default=None, max_length=300)
    transcribed_text: str | None = None
    language: str | None = Field(
        default=None,
        pattern=r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$",
        description="Set only when the language was identified confidently.",
    )
    key_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    dates_mentioned: list[DateMention] = Field(default_factory=list)
    unreadable_markers: list[UnreadableMarker] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_marker_ids(self) -> HandwrittenNoteDocument:
        marker_ids = [marker.marker_id for marker in self.unreadable_markers]
        if len(marker_ids) != len(set(marker_ids)):
            raise ValueError("Unreadable marker IDs must be unique.")
        return self
