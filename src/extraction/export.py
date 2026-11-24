"""Safe JSON serialization for extraction results."""

from __future__ import annotations

import json
from typing import Any

from src.schemas.extraction_result import ExtractionResult

SCHEMA_VERSION = "1.0"


def extraction_result_to_dict(result: ExtractionResult) -> dict[str, Any]:
    """Convert a result to a JSON-safe dictionary without raw document bytes."""
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction": result.model_dump(mode="json"),
    }


def dump_extraction_result(result: ExtractionResult, *, indent: int = 2) -> str:
    """Serialize a result with consistent dates, strings for decimals, and nulls."""
    return json.dumps(
        extraction_result_to_dict(result),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
    )
