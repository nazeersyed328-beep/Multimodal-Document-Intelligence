"""Strict schema for diagrams.

Relationships must reference elements that were actually observed. Ambiguous
visual evidence is recorded in ``ambiguities`` instead of being interpreted.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .common import DocumentBase, DocumentType, StrictModel


class DiagramType(str, Enum):
    """Common diagram categories; unknown remains an explicit option."""

    FLOWCHART = "flowchart"
    MIND_MAP = "mind_map"
    UML = "uml"
    ER_DIAGRAM = "er_diagram"
    ARCHITECTURE = "architecture"
    TIMELINE = "timeline"
    OTHER = "other"
    UNKNOWN = "unknown"


class DiagramElement(StrictModel):
    """A visible node, box, shape, or region in a diagram."""

    element_id: str = Field(min_length=1, max_length=100)
    label: str | None = Field(default=None, max_length=500)
    element_type: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=1000)


class DiagramConnection(StrictModel):
    """A visible relationship between two observed elements."""

    source_id: str = Field(min_length=1, max_length=100)
    target_id: str = Field(min_length=1, max_length=100)
    label: str | None = Field(default=None, max_length=500)
    relationship_type: str | None = Field(default=None, max_length=100)


class Ambiguity(StrictModel):
    """A place where the visual evidence does not support a firm conclusion."""

    description: str = Field(min_length=1, max_length=1000)
    evidence: str | None = Field(default=None, max_length=1000)


class DiagramDocument(DocumentBase):
    """Typed payload for a diagram."""

    document_type: Literal[DocumentType.DIAGRAM] = DocumentType.DIAGRAM
    title: str | None = Field(default=None, max_length=300)
    diagram_type: DiagramType | None = None
    elements: list[DiagramElement] = Field(default_factory=list)
    connections: list[DiagramConnection] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    extracted_text: list[str] = Field(default_factory=list)
    short_description: str | None = Field(default=None, max_length=2000)
    ambiguities: list[Ambiguity] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_diagram_structure(self) -> DiagramDocument:
        element_ids = [element.element_id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("Diagram element IDs must be unique.")
        known_ids = set(element_ids)
        for connection in self.connections:
            if connection.source_id not in known_ids:
                raise ValueError(
                    f"Connection source_id {connection.source_id!r} "
                    "does not reference a known diagram element."
                )
            if connection.target_id not in known_ids:
                raise ValueError(
                    f"Connection target_id {connection.target_id!r} "
                    "does not reference a known diagram element."
                )
        return self
