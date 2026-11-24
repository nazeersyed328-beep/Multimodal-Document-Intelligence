"""Tests for provider configuration, request construction, and safe error mapping."""

from __future__ import annotations

import base64
import logging
from types import SimpleNamespace

import pytest

from src.extraction.gemini_provider import GeminiVisionProvider
from src.extraction.parser import parse_extraction_envelope, parse_json_text
from src.extraction.provider import (
    UNAVAILABLE,
    AuthenticationError,
    InvalidResponseError,
    MissingApiKeyError,
    NetworkError,
    ProviderError,
    QuotaExceededError,
    VisionImage,
    VisionRequest,
)
from src.schemas.common import DocumentType

from .conftest import make_extraction_settings, make_test_settings, make_png


class FakeResponse:
    output_text = '{"document_type":"invoice"}'


class FakeCreate:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.kwargs: dict = {}
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs.copy())
        if self.error is not None:
            raise self.error
        return self.result or FakeResponse()


def install_fake_client(monkeypatch, create: FakeCreate) -> dict:
    created: dict = {}

    def fake_client(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(interactions=SimpleNamespace(create=create))

    monkeypatch.setattr("google.genai.Client", fake_client)
    return created


def request() -> VisionRequest:
    return VisionRequest(
        images=(
            VisionImage(
                data=make_png(),
                mime_type="image/png",
                page_number=1,
                label="test page",
            ),
        ),
        prompt="Extract data",
        response_schema={"type": "object"},
        temperature=0.1,
        max_output_tokens=1234,
        timeout_seconds=17,
    )


def test_missing_api_key_does_not_create_client(monkeypatch) -> None:
    def fail(**kwargs):
        raise AssertionError("Client should not be created without a key.")

    monkeypatch.setattr("google.genai.Client", fail)
    with pytest.raises(MissingApiKeyError):
        GeminiVisionProvider(make_test_settings())


def test_client_uses_model_timeout_and_retry_options(monkeypatch) -> None:
    create = FakeCreate()
    created = install_fake_client(monkeypatch, create)
    settings = make_extraction_settings(
        API_TIMEOUT_SECONDS="33",
        MAX_RETRIES="3",
        RETRY_BACKOFF_SECONDS="2",
        RETRY_MAX_DELAY_SECONDS="20",
    )
    provider = GeminiVisionProvider(settings)

    assert created["api_key"] == "test-key-not-real"
    http_options = created["http_options"]
    assert http_options.timeout == 33
    retries = http_options.retry_options
    assert retries.attempts == 4
    assert retries.initial_delay == 2
    assert retries.max_delay == 20
    assert 429 in retries.http_status_codes

    response = provider.extract(request())
    assert response.text == FakeResponse.output_text
    assert response.provider == "gemini"
    assert response.model == "gemini-3.8-flash"
    assert create.kwargs["model"] == "gemini-3.8-flash"
    assert create.kwargs["timeout"] == 17
    assert create.kwargs["generation_config"] == {
        "temperature": 0.1,
        "max_output_tokens": 1234,
    }
    assert create.kwargs["response_mime_type"] == "application/json"
    assert create.kwargs["response_format"]["type"] == "text"
    assert create.kwargs["response_format"]["mime_type"] == "application/json"
    assert create.kwargs["response_format"]["schema"] == {"type": "object"}
    assert create.kwargs["automatic_continuation"] is False
    assert len(create.calls) == 1

    input_value = create.kwargs["input"]
    assert len(input_value) == 1
    assert input_value[0]["type"] == "user_input"
    parts = input_value[0]["content"]
    assert parts[0]["text"].startswith("Page 1")
    assert parts[1]["type"] == "image"
    assert parts[1]["mime_type"] == "image/png"
    assert base64.b64decode(parts[1]["data"]) == make_png()
    assert parts[-1]["text"] == "Extract data"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (type("QuotaError", (Exception,), {"status_code": 429})(), QuotaExceededError),
        (type("AuthError", (Exception,), {"status_code": 401})(), AuthenticationError),
        (TimeoutError("deadline exceeded"), NetworkError),
        (RuntimeError("unexpected"), ProviderError),
    ],
)
def test_provider_errors_are_sanitized(monkeypatch, error, expected) -> None:
    install_fake_client(monkeypatch, FakeCreate(error=error))
    provider = GeminiVisionProvider(make_extraction_settings())
    with pytest.raises(expected) as exc_info:
        provider.extract(request())
    assert "test-key-not-real" not in str(exc_info.value)
    assert "unexpected" not in str(exc_info.value)


def test_empty_provider_response_is_rejected(monkeypatch) -> None:
    class EmptyResponse:
        output_text = "   "

    install_fake_client(monkeypatch, FakeCreate(result=EmptyResponse()))
    provider = GeminiVisionProvider(make_extraction_settings())
    with pytest.raises(ProviderError, match="empty response"):
        provider.extract(request())


def test_unexpected_provider_response_shape_is_rejected(monkeypatch) -> None:
    class UnknownResponse:
        pass

    install_fake_client(monkeypatch, FakeCreate(result=UnknownResponse()))
    provider = GeminiVisionProvider(make_extraction_settings())
    with pytest.raises(InvalidResponseError, match="unexpected response shape"):
        provider.extract(request())


class FakeBadRequestError(Exception):
    def __init__(self, body=None, headers=None) -> None:
        super().__init__("bad request")
        self.status_code = 400
        self.body = body
        if headers is not None:
            self.response = SimpleNamespace(headers=headers)


def test_bad_request_diagnostics_are_allowlisted_and_sanitized(monkeypatch) -> None:
    api_key = "AIza" + "secretkey0123456789" * 3
    image_payload = "A" * 80
    body = {
        "error": {
            "code": "invalid_request",
            "message": (
                f"Bad response_format.schema api_key={api_key} "
                f"Authorization: Bearer token-{api_key} image_data={image_payload}"
            ),
        },
        "request": {"headers": {"authorization": "Bearer secret-header"}},
        "raw_image": image_payload,
    }
    error = FakeBadRequestError(
        body,
        {"x-goog-api-key": "secret-header", "x-request-id": "req_123"},
    )
    install_fake_client(monkeypatch, FakeCreate(error=error))
    provider = GeminiVisionProvider(make_extraction_settings())

    with pytest.raises(ProviderError) as exc_info:
        provider.extract(request())

    diagnostics = exc_info.value.diagnostics
    assert diagnostics.http_status == 400
    assert diagnostics.sdk_exception_class == "FakeBadRequestError"
    assert diagnostics.provider_error_code == "invalid_request"
    assert diagnostics.request_id == "req_123"
    assert api_key not in diagnostics.provider_error_message
    assert image_payload not in diagnostics.provider_error_message
    assert "redacted" in diagnostics.provider_error_message
    assert len(diagnostics.provider_error_message) <= 303


def test_error_logs_do_not_expose_secret_or_image_payload(monkeypatch, caplog) -> None:
    api_key = "AIza" + "secretkey0123456789" * 3
    image_payload = "B" * 90
    error = FakeBadRequestError(
        {
            "error": {
                "code": "invalid_request",
                "message": f"Authorization: Bearer {api_key} image_data={image_payload}",
            }
        },
        {"x-goog-api-key": "secret-header"},
    )
    install_fake_client(monkeypatch, FakeCreate(error=error))
    provider = GeminiVisionProvider(make_extraction_settings())
    caplog.set_level(
        logging.WARNING,
        logger="multimodal_document_intelligence.extraction.gemini",
    )

    with pytest.raises(ProviderError):
        provider.extract(request())

    assert "test-key-not-real" not in caplog.text
    assert "secret-header" not in caplog.text
    assert api_key not in caplog.text
    assert image_payload not in caplog.text
    assert "Bearer" not in caplog.text


def test_missing_error_fields_are_reported_as_unavailable(monkeypatch) -> None:
    install_fake_client(monkeypatch, FakeCreate(error=RuntimeError("unexpected")))
    provider = GeminiVisionProvider(make_extraction_settings())

    with pytest.raises(ProviderError) as exc_info:
        provider.extract(request())

    diagnostics = exc_info.value.diagnostics
    assert diagnostics.http_status == UNAVAILABLE
    assert diagnostics.sdk_exception_class == "RuntimeError"
    assert diagnostics.provider_error_code == UNAVAILABLE
    assert diagnostics.provider_error_message == UNAVAILABLE
    assert diagnostics.request_id == UNAVAILABLE
    assert str(exc_info.value) == ProviderError.user_message


def test_sdk2_steps_response_is_parsed_from_model_output(monkeypatch) -> None:
    response = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="user_input",
                content=[SimpleNamespace(type="text", text="Extract data")],
            ),
            SimpleNamespace(
                type="thought",
                content=[SimpleNamespace(type="text", text="thinking")],
            ),
            SimpleNamespace(
                type="model_output",
                content=[
                    {"type": "text", "text": '{"document_type":'},
                    SimpleNamespace(type="text", text='"invoice"}'),
                ],
            ),
        ],
        output_text="",
    )
    install_fake_client(monkeypatch, FakeCreate(result=response))
    provider = GeminiVisionProvider(make_extraction_settings())

    result = provider.extract(request())

    assert result.text == '{"document_type":"invoice"}'


def test_sdk2_empty_steps_response_is_rejected(monkeypatch) -> None:
    response = SimpleNamespace(steps=[], output_text="   ")
    install_fake_client(monkeypatch, FakeCreate(result=response))
    provider = GeminiVisionProvider(make_extraction_settings())

    with pytest.raises(ProviderError, match="empty response"):
        provider.extract(request())


def test_sdk2_structured_json_output_passes_pydantic_validation(monkeypatch) -> None:
    payload = (
        '{"document_type":"invoice","data":{"vendor_name":"Acme"},'
        '"missing_fields":[],"field_confidence":[]}'
    )
    response = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text=payload)],
            )
        ]
    )
    install_fake_client(monkeypatch, FakeCreate(result=response))
    provider = GeminiVisionProvider(make_extraction_settings())

    result = provider.extract(request())
    envelope = parse_extraction_envelope(result.text)

    assert envelope.document_type is DocumentType.INVOICE
    assert envelope.data["vendor_name"] == "Acme"


def test_sdk2_invalid_json_output_is_rejected_by_parser(monkeypatch) -> None:
    response = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text="not json")],
            )
        ]
    )
    install_fake_client(monkeypatch, FakeCreate(result=response))
    provider = GeminiVisionProvider(make_extraction_settings())

    result = provider.extract(request())

    with pytest.raises(InvalidResponseError):
        parse_json_text(result.text)


def test_sdk2_pydantic_validation_failure_is_still_reported(monkeypatch) -> None:
    response = SimpleNamespace(
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[
                    SimpleNamespace(
                        type="text",
                        text='{"document_type":"invoice","surprise":1}',
                    )
                ],
            )
        ]
    )
    install_fake_client(monkeypatch, FakeCreate(result=response))
    provider = GeminiVisionProvider(make_extraction_settings())

    result = provider.extract(request())

    with pytest.raises(InvalidResponseError):
        parse_extraction_envelope(result.text)
