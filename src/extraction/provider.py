"""Provider-neutral interface for multimodal vision extraction."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisionImage:
    """One prepared page image sent to a vision provider."""

    data: bytes
    mime_type: str
    page_number: int
    label: str = ""


@dataclass(frozen=True)
class VisionRequest:
    """A complete provider request with JSON schema constraints."""

    images: tuple[VisionImage, ...]
    prompt: str
    response_schema: dict[str, Any]
    temperature: float = 0.0
    max_output_tokens: int = 8192
    timeout_seconds: int = 60


@dataclass(frozen=True)
class VisionResponse:
    """Raw text returned by a provider before parsing."""

    text: str
    provider: str
    model: str


UNAVAILABLE = "unavailable"
_MAX_DIAGNOSTIC_TEXT_LENGTH = 300
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"\b(api[_-]?key|authorization|bearer|x-goog-api-key|token|secret|password)\b"
    r"\s*[:=]\s*['\"]?[^\s,'\"}\]]+",
    re.IGNORECASE,
)
_GOOGLE_API_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{20,}")
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_REQUEST_ID_ALLOWLIST = ("x-request-id", "x-goog-request-id", "request-id")


@dataclass(frozen=True)
class ProviderErrorDiagnostics:
    """Allowlisted, sanitized provider error fields safe for reporting."""

    http_status: int | str = UNAVAILABLE
    sdk_exception_class: str = UNAVAILABLE
    provider_error_code: str = UNAVAILABLE
    provider_error_message: str = UNAVAILABLE
    request_id: str = UNAVAILABLE


def sanitize_diagnostic_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    text = _SENSITIVE_ASSIGNMENT_RE.sub(r"\1=<redacted>", text)
    text = _BEARER_TOKEN_RE.sub("Bearer <redacted>", text)
    text = _GOOGLE_API_KEY_RE.sub("<redacted-api-key>", text)
    text = _BASE64_RE.sub("<redacted-base64>", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) > _MAX_DIAGNOSTIC_TEXT_LENGTH:
        return text[:_MAX_DIAGNOSTIC_TEXT_LENGTH] + "..."
    return text


def sanitize_request_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 120 or not re.fullmatch(r"[A-Za-z0-9._:-]+", text):
        return None
    return text


def safe_http_status(value: Any) -> int | str:
    if isinstance(value, bool) or not isinstance(value, int):
        return UNAVAILABLE
    if not 100 <= value <= 599:
        return UNAVAILABLE
    return value


class ProviderError(Exception):
    """Base class for safe, user-presentable provider failures."""

    code = "provider_error"
    user_message = "The vision provider could not complete the request."

    def __init__(
        self,
        detail: str | None = None,
        *,
        diagnostics: ProviderErrorDiagnostics | None = None,
    ) -> None:
        self.detail = detail
        self.diagnostics = diagnostics or ProviderErrorDiagnostics()
        super().__init__(detail or self.user_message)


class MissingApiKeyError(ProviderError):
    code = "missing_api_key"
    user_message = "GEMINI_API_KEY is not configured. Add it to .env and restart the app."


class AuthenticationError(ProviderError):
    code = "authentication_error"
    user_message = "Gemini authentication failed. Check that the API key is valid and enabled."


class QuotaExceededError(ProviderError):
    code = "quota_exceeded"
    user_message = "Gemini quota or rate limit reached. Wait and try again later."


class NetworkError(ProviderError):
    code = "network_error"
    user_message = "A network or timeout error occurred while contacting Gemini."


class ModelUnavailableError(ProviderError):
    code = "model_unavailable"
    user_message = "The configured Gemini model is unavailable. Check GEMINI_MODEL."


class InvalidResponseError(ProviderError):
    code = "invalid_response"
    user_message = "The vision model returned an empty or unusable response."


class VisionProvider(ABC):
    """Minimal contract implemented by concrete vision providers."""

    name: str

    @abstractmethod
    def extract(self, request: VisionRequest) -> VisionResponse:
        """Send images and a prompt to the provider and return raw text."""


def create_vision_provider(settings: Any) -> VisionProvider:
    """Create the configured provider. Import is local to avoid cycles."""
    provider_name = getattr(settings, "vision_provider", "gemini")
    if provider_name != "gemini":
        raise ProviderError(f"Unsupported vision provider: {provider_name}.")
    from .gemini_provider import create_gemini_provider

    return create_gemini_provider(settings)
