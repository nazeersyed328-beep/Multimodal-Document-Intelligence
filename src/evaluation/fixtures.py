"""Clearly labeled offline fixture providers for pipeline tests."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from src.extraction.provider import VisionProvider, VisionRequest, VisionResponse

from .schemas import GroundTruth

_PATH_PART_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def unflatten_fields(fields: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in fields.items():
        if value is None:
            continue
        tokens: list[str | int] = []
        for name, index in _PATH_PART_RE.findall(path):
            tokens.append(name if name else int(index))
        cursor: Any = root
        for position, token in enumerate(tokens):
            last = position == len(tokens) - 1
            next_token = tokens[position + 1] if not last else None
            if isinstance(token, int):
                while len(cursor) <= token:
                    cursor.append({} if isinstance(next_token, str) else [])
                if last:
                    cursor[token] = value
                else:
                    cursor = cursor[token]
            else:
                if last:
                    cursor[token] = value
                else:
                    container = [] if isinstance(next_token, int) else {}
                    cursor[token] = cursor.get(token, container)
                    cursor = cursor[token]
    return root


class FixtureVisionProvider(VisionProvider):
    """Return deterministic fixture envelopes; never represents a real API result."""

    def __init__(
        self,
        truths: Iterable[GroundTruth],
        *,
        name: str = "fixture",
        confidence: float = 0.99,
    ) -> None:
        self.name = name
        self._truths = list(truths)
        self._confidence = confidence
        self._next_index = 0
        self._current_index = 0

    def _response(self, payload: dict[str, Any]) -> VisionResponse:
        return VisionResponse(
            text=json.dumps(payload, ensure_ascii=False, default=str),
            provider=self.name,
            model="fixture-model",
        )

    def extract(self, request: VisionRequest) -> VisionResponse:
        schema_title = str(request.response_schema.get("title", ""))
        if schema_title == "DocumentTypeDecision":
            self._current_index = min(self._next_index, len(self._truths) - 1)
            self._next_index += 1
            truth = self._truths[self._current_index]
            return self._response(
                {
                    "document_type": truth.expected_document_type.value,
                    "confidence": self._confidence,
                    "reason": "deterministic local fixture",
                }
            )
        if schema_title != "ExtractionEnvelope":
            raise ValueError(f"Fixture provider received unexpected schema: {schema_title}")
        truth = self._truths[self._current_index]
        low_confidence = set(truth.expected_low_confidence_fields)
        confidence_paths = list(truth.expected_fields)
        if (
            truth.expected_document_type.value == "diagram"
            and "elements" not in confidence_paths
        ):
            confidence_paths.append("elements")
        payload = {
            "document_type": truth.expected_document_type.value,
            "data": unflatten_fields(truth.expected_fields),
            "missing_fields": [
                path for path, value in truth.expected_fields.items() if value is None
            ],
            "unreadable_fields": list(truth.expected_unreadable_fields),
            "warnings": [],
            "overall_confidence": self._confidence,
            "confidence_note": "Confidence values are model estimates, not calibrated probabilities.",
            "field_confidence": [
                {
                    "field_path": path,
                    "confidence": 0.42 if path in low_confidence else self._confidence,
                    "evidence": "fixture low-confidence marker" if path in low_confidence else None,
                    "needs_review": path in low_confidence,
                }
                for path in confidence_paths
            ],
        }
        return self._response(payload)
