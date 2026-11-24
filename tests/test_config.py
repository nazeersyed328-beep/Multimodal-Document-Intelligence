"""Unit tests for environment-based configuration."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.config import ConfigurationError, load_settings


def test_load_settings_with_empty_environment() -> None:
    settings = load_settings(environ={})
    assert settings.gemini_api_key is None
    assert settings.is_api_key_configured is False
    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.max_upload_size_mb == 20
    assert settings.allowed_file_extensions == (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    )
    assert settings.log_level == "INFO"
    assert settings.confidence_threshold == 0.75
    assert settings.api_timeout_seconds == 60
    assert settings.env_file_loaded is False


def test_load_settings_reads_configured_values() -> None:
    settings = load_settings(
        environ={
            "GEMINI_API_KEY": "real-looking-key",
            "GEMINI_MODEL": "gemini-test-model",
            "MAX_UPLOAD_SIZE_MB": "5",
            "ALLOWED_FILE_EXTENSIONS": ".PDF, .png",
            "LOG_LEVEL": "debug",
            "CONFIDENCE_THRESHOLD": "0.9",
            "API_TIMEOUT_SECONDS": "30",
        }
    )
    assert settings.gemini_api_key == "real-looking-key"
    assert settings.is_api_key_configured is True
    assert settings.gemini_model == "gemini-test-model"
    assert settings.max_upload_size_mb == 5
    assert settings.allowed_file_extensions == (".pdf", ".png")
    assert settings.log_level == "DEBUG"
    assert settings.confidence_threshold == 0.9
    assert settings.api_timeout_seconds == 30


def test_placeholder_api_key_treated_as_missing() -> None:
    settings = load_settings(environ={"GEMINI_API_KEY": "your-api-key-here"})
    assert settings.gemini_api_key is None
    assert settings.is_api_key_configured is False


def test_missing_api_key_does_not_crash() -> None:
    settings = load_settings(environ={})
    assert settings.public_summary()["gemini_api_key"] == "not configured"


def test_public_summary_never_contains_api_key() -> None:
    secret = "super-secret-key-value"
    settings = load_settings(environ={"GEMINI_API_KEY": secret})
    summary_text = str(settings.public_summary())
    assert secret not in summary_text
    assert "configured" in summary_text


def test_max_upload_size_bytes() -> None:
    settings = load_settings(environ={"MAX_UPLOAD_SIZE_MB": "2"})
    assert settings.max_upload_size_bytes == 2 * 1024 * 1024


def test_output_dir_resolves_relative_to_project_root() -> None:
    settings = load_settings(environ={"OUTPUT_DIR": "custom_outputs"})
    assert settings.output_dir.is_absolute()
    assert settings.output_dir.name == "custom_outputs"


def test_invalid_confidence_threshold_rejected() -> None:
    with pytest.raises(ConfigurationError, match="between 0 and 1"):
        load_settings(environ={"CONFIDENCE_THRESHOLD": "1.5"})


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ConfigurationError, match="LOG_LEVEL"):
        load_settings(environ={"LOG_LEVEL": "VERBOSE"})


def test_invalid_extension_rejected() -> None:
    with pytest.raises(ConfigurationError, match="ALLOWED_FILE_EXTENSIONS"):
        load_settings(environ={"ALLOWED_FILE_EXTENSIONS": "pdf,png"})


def test_nonpositive_upload_limit_rejected() -> None:
    with pytest.raises(ConfigurationError, match="greater than zero"):
        load_settings(environ={"MAX_UPLOAD_SIZE_MB": "0"})


def test_load_settings_reads_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_MODEL=gemini-from-file\nLOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {}, clear=True):
        settings = load_settings(env_file=env_file)
    assert settings.env_file_loaded is True
    assert settings.gemini_model == "gemini-from-file"
    assert settings.log_level == "WARNING"


def test_missing_env_file_does_not_crash(tmp_path) -> None:
    with patch.dict(os.environ, {}, clear=True):
        settings = load_settings(env_file=tmp_path / "absent.env")
    assert settings.env_file_loaded is False
    assert settings.gemini_api_key is None


def test_phase2_defaults_are_configured() -> None:
    settings = load_settings(environ={})
    assert settings.max_pages_per_pdf == 50
    assert settings.max_pages_to_process == 10
    assert settings.pdf_render_dpi == 110.0
    assert settings.max_pdf_render_pixels == 20_000_000
    assert settings.max_image_pixels == 25_000_000
    assert settings.preview_max_dimension == 900
    assert settings.vision_max_dimension == 1536
    assert settings.max_documents_per_session == 20
    assert settings.preserve_exif_orientation is True


def test_phase2_settings_load_from_environment() -> None:
    settings = load_settings(
        environ={
            "MAX_PAGES_PER_PDF": "12",
            "MAX_PAGES_TO_PROCESS": "4",
            "PDF_RENDER_DPI": "144.5",
            "MAX_PDF_RENDER_PIXELS": "5000000",
            "MAX_IMAGE_PIXELS": "6000000",
            "PREVIEW_MAX_DIMENSION": "640",
            "VISION_MAX_DIMENSION": "1024",
            "MAX_DOCUMENTS_PER_SESSION": "8",
            "PRESERVE_EXIF_ORIENTATION": "no",
        }
    )
    assert settings.max_pages_per_pdf == 12
    assert settings.max_pages_to_process == 4
    assert settings.pdf_render_dpi == 144.5
    assert settings.max_pdf_render_pixels == 5_000_000
    assert settings.max_image_pixels == 6_000_000
    assert settings.preview_max_dimension == 640
    assert settings.vision_max_dimension == 1024
    assert settings.max_documents_per_session == 8
    assert settings.preserve_exif_orientation is False


def test_invalid_render_dpi_rejected() -> None:
    with pytest.raises(ConfigurationError, match="PDF_RENDER_DPI"):
        load_settings(environ={"PDF_RENDER_DPI": "0"})


def test_invalid_boolean_setting_rejected() -> None:
    with pytest.raises(ConfigurationError, match="PRESERVE_EXIF_ORIENTATION"):
        load_settings(environ={"PRESERVE_EXIF_ORIENTATION": "maybe"})


def test_public_summary_includes_limits_but_no_key() -> None:
    summary = load_settings(environ={"GEMINI_API_KEY": "secret-should-not-appear"}).public_summary()
    assert "secret-should-not-appear" not in str(summary)
    assert summary["max_pages_per_pdf"] == "50"
    assert summary["vision_max_dimension"] == "1536"


def test_phase3_defaults_are_configured() -> None:
    settings = load_settings(environ={})
    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.vision_provider == "gemini"
    assert settings.max_retries == 2
    assert settings.retry_backoff_seconds == 1.0
    assert settings.retry_max_delay_seconds == 60.0
    assert settings.temperature == 0.0
    assert settings.max_output_tokens == 8192
    assert settings.confidence_review_threshold == 0.75


def test_phase3_settings_load_from_environment() -> None:
    settings = load_settings(
        environ={
            "VISION_PROVIDER": "gemini",
            "MAX_RETRIES": "4",
            "RETRY_BACKOFF_SECONDS": "2.5",
            "RETRY_MAX_DELAY_SECONDS": "12",
            "TEMPERATURE": "0.2",
            "MAX_OUTPUT_TOKENS": "2048",
            "CONFIDENCE_REVIEW_THRESHOLD": "0.8",
        }
    )
    assert settings.vision_provider == "gemini"
    assert settings.max_retries == 4
    assert settings.retry_backoff_seconds == 2.5
    assert settings.retry_max_delay_seconds == 12.0
    assert settings.temperature == 0.2
    assert settings.max_output_tokens == 2048
    assert settings.confidence_review_threshold == 0.8


def test_confidence_review_threshold_falls_back_to_legacy_setting() -> None:
    settings = load_settings(environ={"CONFIDENCE_THRESHOLD": "0.6"})
    assert settings.confidence_review_threshold == 0.6


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VISION_PROVIDER", "openai"),
        ("MAX_RETRIES", "-1"),
        ("RETRY_BACKOFF_SECONDS", "-1"),
        ("RETRY_MAX_DELAY_SECONDS", "0"),
        ("TEMPERATURE", "-0.1"),
        ("TEMPERATURE", "2.1"),
        ("MAX_OUTPUT_TOKENS", "0"),
        ("CONFIDENCE_REVIEW_THRESHOLD", "2"),
    ],
)
def test_invalid_phase3_settings_rejected(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        load_settings(environ={name: value})


def test_public_summary_includes_phase3_settings_but_no_key() -> None:
    summary = load_settings(environ={"GEMINI_API_KEY": "secret-should-not-appear"}).public_summary()
    assert "secret-should-not-appear" not in str(summary)
    assert summary["vision_provider"] == "gemini"
    assert summary["gemini_model"] == "gemini-3.8-flash"
    assert summary["max_retries"] == "2"
    assert summary["confidence_review_threshold"] == "0.75"


def test_phase4_defaults_are_configured() -> None:
    settings = load_settings(environ={})
    assert settings.amount_tolerance == Decimal("0.01")
    assert settings.future_date_tolerance_days == 365
    assert settings.max_past_date_days == 3650
    assert settings.max_quality_issues == 200


def test_phase4_settings_load_from_environment() -> None:
    settings = load_settings(
        environ={
            "AMOUNT_TOLERANCE": "0.05",
            "FUTURE_DATE_TOLERANCE_DAYS": "90",
            "MAX_PAST_DATE_DAYS": "730",
            "MAX_QUALITY_ISSUES": "50",
        }
    )
    assert settings.amount_tolerance == Decimal("0.05")
    assert settings.future_date_tolerance_days == 90
    assert settings.max_past_date_days == 730
    assert settings.max_quality_issues == 50


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AMOUNT_TOLERANCE", "-0.01"),
        ("FUTURE_DATE_TOLERANCE_DAYS", "-1"),
        ("MAX_PAST_DATE_DAYS", "-1"),
        ("MAX_QUALITY_ISSUES", "0"),
    ],
)
def test_invalid_phase4_settings_rejected(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=name):
        load_settings(environ={name: value})


def test_public_summary_includes_phase4_settings_but_no_key() -> None:
    summary = load_settings(environ={"GEMINI_API_KEY": "secret-should-not-appear"}).public_summary()
    assert "secret-should-not-appear" not in str(summary)
    assert summary["amount_tolerance"] == "0.01"
    assert summary["future_date_tolerance_days"] == "365"
    assert summary["max_past_date_days"] == "3650"
    assert summary["max_quality_issues"] == "200"
