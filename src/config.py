"""Environment-driven configuration for the document-intelligence pipeline.

No API client is created here and no network request is ever made. A missing
API key is a valid state: ingestion, preprocessing, and previews still work
locally and the application reports that the key is not configured.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)
VALID_LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)
VALID_VISION_PROVIDERS = frozenset({"gemini"})
PLACEHOLDER_API_KEYS = frozenset(
    {
        "",
        "your-api-key-here",
        "your_api_key_here",
        "your-api-key",
        "changeme",
        "change-me",
        "placeholder",
        "replace-me",
        "none",
    }
)


class ConfigurationError(ValueError):
    """Raised when an environment value cannot be parsed into a valid setting."""


def _read(name: str, environ: Mapping[str, str], default: str) -> str:
    value = environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _parse_positive_int(name: str, raw: str, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer; got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero; got {value}.")
    return value


def _parse_confidence_threshold(raw: str, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"CONFIDENCE_THRESHOLD must be a number between 0 and 1; got {raw!r}."
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(
            f"CONFIDENCE_THRESHOLD must be between 0 and 1; got {value}."
        )
    return value


def _parse_nonnegative_int(name: str, raw: str, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer; got {raw!r}.") from exc
    if value < 0:
        raise ConfigurationError(f"{name} cannot be negative; got {value}.")
    return value


def _parse_bounded_float(
    name: str,
    raw: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{name} must be a number between {minimum} and {maximum}; got {raw!r}."
        ) from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum}; got {value}."
        )
    return value


def _parse_positive_float(name: str, raw: str, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number; got {raw!r}.") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero; got {value}.")
    return value


def _parse_nonnegative_float(name: str, raw: str, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number; got {raw!r}.") from exc
    if not value >= 0:
        raise ConfigurationError(f"{name} cannot be negative; got {value}.")
    return value


def _parse_nonnegative_decimal(name: str, raw: str, default: Decimal) -> Decimal:
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a decimal number; got {raw!r}.") from exc
    if not value.is_finite() or value < 0:
        raise ConfigurationError(
            f"{name} must be a finite nonnegative decimal; got {raw!r}."
        )
    return value


def _parse_bool(name: str, raw: str, default: bool) -> bool:
    normalized = raw.strip().lower()
    true_values = {"1", "true", "yes", "on"}
    false_values = {"0", "false", "no", "off"}
    if normalized in true_values:
        return True
    if normalized in false_values:
        return False
    raise ConfigurationError(
        f"{name} must be one of true/false/yes/no/1/0; got {raw!r}."
    )


def _parse_log_level(raw: str, default: str) -> str:
    level = raw.upper()
    if level not in VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigurationError(
            f"LOG_LEVEL must be one of: {allowed}; got {raw!r}."
        )
    return level


def _parse_vision_provider(raw: str, default: str) -> str:
    provider = raw.strip().lower()
    if provider not in VALID_VISION_PROVIDERS:
        allowed = ", ".join(sorted(VALID_VISION_PROVIDERS))
        raise ConfigurationError(
            f"VISION_PROVIDER must be one of: {allowed}; got {raw!r}."
        )
    return provider


def _parse_extensions(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    cleaned: list[str] = []
    for item in items:
        extension = item.strip().lower()
        if not extension:
            continue
        if (
            not extension.startswith(".")
            or len(extension) < 2
            or not extension[1:].isalnum()
        ):
            raise ConfigurationError(
                "ALLOWED_FILE_EXTENSIONS entries must look like '.pdf' or '.png'; "
                f"got {item!r}."
            )
        if extension not in cleaned:
            cleaned.append(extension)
    if not cleaned:
        raise ConfigurationError(
            "ALLOWED_FILE_EXTENSIONS must contain at least one file extension."
        )
    return tuple(cleaned)


@dataclass(frozen=True)
class Settings:
    """Validated, immutable configuration values."""

    gemini_api_key: str | None
    gemini_model: str
    max_upload_size_mb: int
    allowed_file_extensions: tuple[str, ...]
    log_level: str
    confidence_threshold: float
    output_dir: Path
    api_timeout_seconds: int
    vision_provider: str = "gemini"
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    retry_max_delay_seconds: float = 60.0
    temperature: float = 0.0
    max_output_tokens: int = 8192
    confidence_review_threshold: float = 0.75
    amount_tolerance: Decimal = Decimal("0.01")
    future_date_tolerance_days: int = 365
    max_past_date_days: int = 3650
    max_quality_issues: int = 200
    evaluation_max_workers: int = 1
    evaluation_request_delay_seconds: float = 1.0
    max_pages_per_pdf: int = 50
    max_pages_to_process: int = 10
    pdf_render_dpi: float = 110.0
    max_pdf_render_pixels: int = 20_000_000
    max_image_pixels: int = 25_000_000
    preview_max_dimension: int = 900
    vision_max_dimension: int = 1536
    max_documents_per_session: int = 20
    preserve_exif_orientation: bool = True
    env_file_loaded: bool = False

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size expressed in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_api_key_configured(self) -> bool:
        """Whether a real-looking API key was supplied."""
        return self.gemini_api_key is not None

    def public_summary(self) -> dict[str, str]:
        """Return a display-safe summary that never includes the API key."""
        return {
            "gemini_api_key": (
                "configured" if self.is_api_key_configured else "not configured"
            ),
            "gemini_model": self.gemini_model,
            "max_upload_size_mb": str(self.max_upload_size_mb),
            "allowed_file_extensions": ", ".join(self.allowed_file_extensions),
            "log_level": self.log_level,
            "confidence_threshold": f"{self.confidence_threshold:.2f}",
            "output_dir": str(self.output_dir),
            "api_timeout_seconds": str(self.api_timeout_seconds),
            "vision_provider": self.vision_provider,
            "max_retries": str(self.max_retries),
            "retry_backoff_seconds": f"{self.retry_backoff_seconds:g}",
            "temperature": f"{self.temperature:g}",
            "max_output_tokens": str(self.max_output_tokens),
            "confidence_review_threshold": f"{self.confidence_review_threshold:.2f}",
            "amount_tolerance": str(self.amount_tolerance),
            "future_date_tolerance_days": str(self.future_date_tolerance_days),
            "max_past_date_days": str(self.max_past_date_days),
            "max_quality_issues": str(self.max_quality_issues),
            "evaluation_max_workers": str(self.evaluation_max_workers),
            "evaluation_request_delay_seconds": f"{self.evaluation_request_delay_seconds:g}",
            "max_pages_per_pdf": str(self.max_pages_per_pdf),
            "max_pages_to_process": str(self.max_pages_to_process),
            "pdf_render_dpi": f"{self.pdf_render_dpi:g}",
            "preview_max_dimension": str(self.preview_max_dimension),
            "vision_max_dimension": str(self.vision_max_dimension),
            "max_documents_per_session": str(self.max_documents_per_session),
            "env_file_loaded": "yes" if self.env_file_loaded else "no",
        }


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> Settings:
    """Build :class:`Settings` from an environment mapping or the real environment.

    Passing ``environ`` keeps the function hermetic for tests and skips loading
    a ``.env`` file entirely.
    """
    env_file_loaded = False
    if environ is None:
        resolved_env_file = (
            Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        )
        env_file_loaded = bool(
            load_dotenv(dotenv_path=resolved_env_file, override=False)
        )
        environ = os.environ

    raw_api_key = _read("GEMINI_API_KEY", environ, "")
    gemini_api_key = (
        None if raw_api_key.lower() in PLACEHOLDER_API_KEYS else raw_api_key
    )

    raw_output_dir = _read("OUTPUT_DIR", environ, "outputs")
    output_dir = Path(raw_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return Settings(
        gemini_api_key=gemini_api_key,
        gemini_model=_read("GEMINI_MODEL", environ, "gemini-3.8-flash"),
        max_upload_size_mb=_parse_positive_int(
            "MAX_UPLOAD_SIZE_MB", _read("MAX_UPLOAD_SIZE_MB", environ, "20"), 20
        ),
        allowed_file_extensions=_parse_extensions(
            _read(
                "ALLOWED_FILE_EXTENSIONS",
                environ,
                ",".join(DEFAULT_ALLOWED_EXTENSIONS),
            ),
            DEFAULT_ALLOWED_EXTENSIONS,
        ),
        log_level=_parse_log_level(_read("LOG_LEVEL", environ, "INFO"), "INFO"),
        confidence_threshold=_parse_confidence_threshold(
            _read("CONFIDENCE_THRESHOLD", environ, "0.75"), 0.75
        ),
        output_dir=output_dir,
        api_timeout_seconds=_parse_positive_int(
            "API_TIMEOUT_SECONDS", _read("API_TIMEOUT_SECONDS", environ, "60"), 60
        ),
        vision_provider=_parse_vision_provider(
            _read("VISION_PROVIDER", environ, "gemini"), "gemini"
        ),
        max_retries=_parse_nonnegative_int(
            "MAX_RETRIES", _read("MAX_RETRIES", environ, "2"), 2
        ),
        retry_backoff_seconds=_parse_positive_float(
            "RETRY_BACKOFF_SECONDS",
            _read("RETRY_BACKOFF_SECONDS", environ, "1"),
            1.0,
        ),
        retry_max_delay_seconds=_parse_positive_float(
            "RETRY_MAX_DELAY_SECONDS",
            _read("RETRY_MAX_DELAY_SECONDS", environ, "60"),
            60.0,
        ),
        temperature=_parse_bounded_float(
            "TEMPERATURE", _read("TEMPERATURE", environ, "0"), 0.0, 0.0, 2.0
        ),
        max_output_tokens=_parse_positive_int(
            "MAX_OUTPUT_TOKENS",
            _read("MAX_OUTPUT_TOKENS", environ, "8192"),
            8192,
        ),
        confidence_review_threshold=_parse_bounded_float(
            "CONFIDENCE_REVIEW_THRESHOLD",
            _read(
                "CONFIDENCE_REVIEW_THRESHOLD",
                environ,
                _read("CONFIDENCE_THRESHOLD", environ, "0.75"),
            ),
            0.0,
            0.0,
            1.0,
        ),
        amount_tolerance=_parse_nonnegative_decimal(
            "AMOUNT_TOLERANCE", _read("AMOUNT_TOLERANCE", environ, "0.01"), Decimal("0.01")
        ),
        future_date_tolerance_days=_parse_nonnegative_int(
            "FUTURE_DATE_TOLERANCE_DAYS",
            _read("FUTURE_DATE_TOLERANCE_DAYS", environ, "365"),
            365,
        ),
        max_past_date_days=_parse_nonnegative_int(
            "MAX_PAST_DATE_DAYS",
            _read("MAX_PAST_DATE_DAYS", environ, "3650"),
            3650,
        ),
        max_quality_issues=_parse_positive_int(
            "MAX_QUALITY_ISSUES",
            _read("MAX_QUALITY_ISSUES", environ, "200"),
            200,
        ),
        evaluation_max_workers=_parse_positive_int(
            "EVALUATION_MAX_WORKERS",
            _read("EVALUATION_MAX_WORKERS", environ, "1"),
            1,
        ),
        evaluation_request_delay_seconds=_parse_nonnegative_float(
            "EVALUATION_REQUEST_DELAY_SECONDS",
            _read("EVALUATION_REQUEST_DELAY_SECONDS", environ, "1"),
            1.0,
        ),
        max_pages_per_pdf=_parse_positive_int(
            "MAX_PAGES_PER_PDF", _read("MAX_PAGES_PER_PDF", environ, "50"), 50
        ),
        max_pages_to_process=_parse_positive_int(
            "MAX_PAGES_TO_PROCESS", _read("MAX_PAGES_TO_PROCESS", environ, "10"), 10
        ),
        pdf_render_dpi=_parse_positive_float(
            "PDF_RENDER_DPI", _read("PDF_RENDER_DPI", environ, "110"), 110.0
        ),
        max_pdf_render_pixels=_parse_positive_int(
            "MAX_PDF_RENDER_PIXELS",
            _read("MAX_PDF_RENDER_PIXELS", environ, "20000000"),
            20_000_000,
        ),
        max_image_pixels=_parse_positive_int(
            "MAX_IMAGE_PIXELS",
            _read("MAX_IMAGE_PIXELS", environ, "25000000"),
            25_000_000,
        ),
        preview_max_dimension=_parse_positive_int(
            "PREVIEW_MAX_DIMENSION",
            _read("PREVIEW_MAX_DIMENSION", environ, "900"),
            900,
        ),
        vision_max_dimension=_parse_positive_int(
            "VISION_MAX_DIMENSION",
            _read("VISION_MAX_DIMENSION", environ, "1536"),
            1536,
        ),
        max_documents_per_session=_parse_positive_int(
            "MAX_DOCUMENTS_PER_SESSION",
            _read("MAX_DOCUMENTS_PER_SESSION", environ, "20"),
            20,
        ),
        preserve_exif_orientation=_parse_bool(
            "PRESERVE_EXIF_ORIENTATION",
            _read("PRESERVE_EXIF_ORIENTATION", environ, "true"),
            True,
        ),
        env_file_loaded=env_file_loaded,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings instance."""
    return load_settings()
