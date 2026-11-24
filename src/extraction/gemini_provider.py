"""Gemini implementation of the provider-neutral vision interface."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from src.config import Settings
from src.utils.logging_config import get_logger

from .provider import (
    UNAVAILABLE,
    AuthenticationError,
    InvalidResponseError,
    MissingApiKeyError,
    ModelUnavailableError,
    NetworkError,
    ProviderError,
    ProviderErrorDiagnostics,
    QuotaExceededError,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    sanitize_diagnostic_text,
    sanitize_request_id,
    safe_http_status,
)

logger = get_logger("extraction.gemini")

_RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]


def _safe_error_kind(exc: Exception) -> tuple[str, str]:
    """Map an SDK exception to a safe code without echoing response content."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    status_text = str(status).lower()
    name = type(exc).__name__.lower()
    combined = f"{status_text} {name}"
    if any(token in combined for token in ("429", "quota", "resource_exhausted")):
        return "quota", QuotaExceededError.user_message
    if any(token in combined for token in ("401", "403", "auth", "permission", "api_key")):
        return "auth", AuthenticationError.user_message
    if any(token in combined for token in ("404", "not_found", "model")):
        return "model", ModelUnavailableError.user_message
    if any(token in combined for token in ("408", "429", "500", "502", "503", "504", "timeout", "deadline", "connection", "network")):
        return "network", NetworkError.user_message
    return "provider", ProviderError.user_message


def _body_value(body: Any, *path: str) -> Any:
    current = body
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _safe_request_id(exc: Exception, body: Any) -> str:
    candidates = [_body_value(body, "request_id"), _body_value(body, "requestId")]
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    get_header = getattr(headers, "get", None)
    if callable(get_header):
        for name in ("x-request-id", "x-goog-request-id"):
            candidates.append(get_header(name))
    for candidate in candidates:
        safe = sanitize_request_id(candidate)
        if safe:
            return safe
    return UNAVAILABLE


def _safe_provider_error_diagnostics(exc: Exception) -> ProviderErrorDiagnostics:
    body = getattr(exc, "body", None)
    code = _body_value(body, "error", "code") or _body_value(body, "code")
    message = _body_value(body, "error", "message") or _body_value(body, "message")
    return ProviderErrorDiagnostics(
        http_status=safe_http_status(getattr(exc, "status_code", None)),
        sdk_exception_class=sanitize_diagnostic_text(type(exc).__name__) or UNAVAILABLE,
        provider_error_code=sanitize_diagnostic_text(code) or UNAVAILABLE,
        provider_error_message=sanitize_diagnostic_text(message) or UNAVAILABLE,
        request_id=_safe_request_id(exc, body),
    )


def _raise_provider_error(exc: Exception) -> None:
    code, message = _safe_error_kind(exc)
    diagnostics = _safe_provider_error_diagnostics(exc)
    logger.warning(
        "Gemini request failed (kind=%s, sdk_error=%s, http_status=%s, provider_code=%s, request_id=%s).",
        code,
        diagnostics.sdk_exception_class,
        diagnostics.http_status,
        diagnostics.provider_error_code,
        diagnostics.request_id,
    )
    if code == "quota":
        raise QuotaExceededError(message, diagnostics=diagnostics) from None
    if code == "auth":
        raise AuthenticationError(message, diagnostics=diagnostics) from None
    if code == "model":
        raise ModelUnavailableError(message, diagnostics=diagnostics) from None
    if code == "network":
        raise NetworkError(message, diagnostics=diagnostics) from None
    raise ProviderError(message, diagnostics=diagnostics) from None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _has_known_response_shape(response: Any) -> bool:
    return any(
        _field(response, name) is not None
        for name in ("steps", "output_text", "text", "candidates")
    )


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for step in _field(response, "steps") or ():
        step_type = _field(step, "type")
        if step_type in ("user_input", "thought"):
            continue
        if step_type not in (None, "model_output"):
            continue
        for content in _field(step, "content") or ():
            content_type = _field(content, "type")
            if content_type not in (None, "text"):
                continue
            candidate = _field(content, "text")
            if isinstance(candidate, str) and candidate.strip():
                parts.append(candidate)
    if parts:
        return "".join(parts)
    fallback = _field(response, "output_text")
    return fallback if isinstance(fallback, str) and fallback.strip() else ""


class GeminiVisionProvider(VisionProvider):
    """Vision provider backed by the official `google-genai` SDK."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.is_api_key_configured or not settings.gemini_api_key:
            raise MissingApiKeyError()

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderError(
                "The google-genai package is not installed. Install requirements.txt."
            ) from exc

        self._settings = settings
        self._types = types
        retry_options = types.HttpRetryOptions(
            attempts=settings.max_retries + 1,
            initial_delay=settings.retry_backoff_seconds,
            max_delay=settings.retry_max_delay_seconds,
            exp_base=2.0,
            jitter=0.1,
            http_status_codes=_RETRYABLE_STATUS_CODES,
        )
        http_options = types.HttpOptions(
            timeout=settings.api_timeout_seconds,
            retry_options=retry_options,
        )
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=http_options,
        )

    def extract(self, request: VisionRequest) -> VisionResponse:
        """Send prepared page images and a prompt with structured JSON output."""
        input_parts: list[dict[str, Any]] = []
        for image in request.images:
            input_parts.append(
                {
                    "type": "text",
                    "text": f"Page {image.page_number}: {image.label or 'document page'}",
                }
            )
            input_parts.append(
                {
                    "type": "image",
                    "data": base64.b64encode(image.data).decode("ascii"),
                    "mime_type": image.mime_type,
                }
            )
        input_parts.append({"type": "text", "text": request.prompt})

        try:
            response = self._client.interactions.create(
                model=self._settings.gemini_model,
                input=[{"type": "user_input", "content": input_parts}],
                generation_config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_output_tokens,
                },
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": request.response_schema,
                },
                response_mime_type="application/json",
                automatic_continuation=False,
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            _raise_provider_error(exc)

        if not _has_known_response_shape(response):
            raise InvalidResponseError(
                "The vision provider returned an unexpected response shape."
            )
        text = _response_text(response)
        if not text.strip():
            raise InvalidResponseError("Gemini returned an empty response.")
        return VisionResponse(
            text=text,
            provider=self.name,
            model=self._settings.gemini_model,
        )


def create_gemini_provider(settings: Settings) -> GeminiVisionProvider:
    """Factory kept separate so service code can stay provider-agnostic."""
    return GeminiVisionProvider(settings)
