"""Phase 3 vision extraction package.

The modules here separate provider integration, response parsing, orchestration,
and export so another vision provider can be added without rewriting the app.
"""

from .export import dump_extraction_result, extraction_result_to_dict
from .gemini_provider import GeminiVisionProvider
from .parser import (
    parse_extraction_envelope,
    parse_json_text,
    parse_typed_document,
)
from .provider import (
    AuthenticationError,
    InvalidResponseError,
    MissingApiKeyError,
    ModelUnavailableError,
    NetworkError,
    ProviderError,
    QuotaExceededError,
    VisionImage,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    create_vision_provider,
)
from .service import ExtractionService

__all__ = [
    "AuthenticationError",
    "ExtractionService",
    "GeminiVisionProvider",
    "InvalidResponseError",
    "MissingApiKeyError",
    "ModelUnavailableError",
    "NetworkError",
    "ProviderError",
    "QuotaExceededError",
    "VisionImage",
    "VisionProvider",
    "VisionRequest",
    "VisionResponse",
    "create_vision_provider",
    "dump_extraction_result",
    "extraction_result_to_dict",
    "parse_extraction_envelope",
    "parse_json_text",
    "parse_typed_document",
]
