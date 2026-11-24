"""Orchestration for document type detection, extraction, and validation."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from src.config import Settings
from src.document.ingestion import DocumentRecord, new_document_id
from src.schemas.common import (
    DocumentType,
    ErrorDetail,
    ErrorSeverity,
    FieldConfidenceEntry,
    ValidationStatus,
)
from src.schemas.document_ingestion import PreprocessingResult
from src.schemas.extraction import DocumentTypeDecision, ExtractionEnvelope
from src.schemas.extraction_result import DocumentPayload, ExtractionResult, ProcessingStatus

from .parser import (
    parse_extraction_envelope,
    parse_json_text,
    parse_typed_document,
    validation_error_details,
    validation_error_messages,
)
from .prompts import (
    build_correction_prompt,
    build_detection_prompt,
    build_extraction_prompt,
)
from .provider import (
    InvalidResponseError,
    ProviderError,
    VisionImage,
    VisionProvider,
    VisionRequest,
    create_vision_provider,
)

_CONFIDENCE_NOTE = "Confidence values are model estimates, not calibrated probabilities."


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _issue(
    code: str,
    message: str,
    field_path: str | None = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
) -> ErrorDetail:
    return ErrorDetail(
        code=code,
        message=message[:2000],
        field_path=field_path[:500] if field_path else None,
        severity=severity,
    )


def _pages_to_images(processed: PreprocessingResult) -> tuple[VisionImage, ...]:
    images: list[VisionImage] = []
    for page in processed.pages:
        if not page.image_bytes or page.processing_status is not ProcessingStatus.COMPLETED:
            continue
        images.append(
            VisionImage(
                data=page.image_bytes,
                mime_type="image/png",
                page_number=page.page_number,
                label=f"{page.original_width}x{page.original_height} prepared page",
            )
        )
    return tuple(images)


def _lookup_value(payload: DocumentPayload, field_path: str) -> Any:
    value: Any = payload.model_dump(mode="python")
    for part in field_path.split("."):
        if isinstance(value, dict):
            if part.isdigit() and part in value:
                value = value[part]
            elif part in value:
                value = value[part]
            else:
                return None
        elif isinstance(value, list) and part.isdigit():
            index = int(part)
            value = value[index] if 0 <= index < len(value) else None
        else:
            return None
    if isinstance(value, (str, int, bool, Decimal)):
        return value
    return None


def _close_to_zero(value: Decimal) -> bool:
    return abs(value) <= Decimal("0.01")


def _arithmetic_warnings(payload: DocumentPayload) -> list[str]:
    warnings: list[str] = []
    if payload.document_type is DocumentType.INVOICE:
        amounts = [item.amount for item in payload.line_items if item.amount is not None]
        if payload.subtotal is not None and amounts and len(amounts) == len(payload.line_items):
            if not _close_to_zero(sum(amounts, Decimal("0")) - payload.subtotal):
                warnings.append("Invoice line-item amounts do not match the printed subtotal.")
        if payload.total is not None and payload.subtotal is not None:
            expected = payload.subtotal + (payload.tax or Decimal("0")) - (
                payload.discount or Decimal("0")
            )
            if not _close_to_zero(expected - payload.total):
                warnings.append("Invoice subtotal, tax, discount, and total are inconsistent.")
    elif payload.document_type is DocumentType.RECEIPT:
        amounts = [item.item_amount for item in payload.items if item.item_amount is not None]
        if payload.total is not None and amounts and len(amounts) == len(payload.items):
            expected = sum(amounts, Decimal("0")) + (payload.tax or Decimal("0"))
            if not _close_to_zero(expected - payload.total):
                warnings.append("Receipt item amounts and tax do not match the printed total.")
    return warnings


class ExtractionService:
    """Coordinate Gemini calls, schema validation, and result construction."""

    def __init__(self, settings: Settings, provider: VisionProvider | None = None) -> None:
        self.settings = settings
        self._provider = provider
        self._provider_error: ProviderError | None = None

    @property
    def provider(self) -> VisionProvider:
        if self._provider is None:
            if self._provider_error is not None:
                raise self._provider_error
            try:
                self._provider = create_vision_provider(self.settings)
            except ProviderError as exc:
                self._provider_error = exc
                raise
        return self._provider

    def _request(
        self,
        images: tuple[VisionImage, ...],
        prompt: str,
        schema: dict[str, Any],
    ) -> VisionRequest:
        return VisionRequest(
            images=images,
            prompt=prompt,
            response_schema=schema,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            timeout_seconds=self.settings.api_timeout_seconds,
        )

    def _detect_document_type(self, images: tuple[VisionImage, ...]) -> DocumentTypeDecision:
        prompt = build_detection_prompt(len(images))
        schema = DocumentTypeDecision.model_json_schema()
        last_error: InvalidResponseError | None = None
        for attempt in range(self.settings.max_retries + 1):
            if attempt == 0:
                current_prompt = prompt
            else:
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was not a valid document type decision. "
                    "Return one JSON object that matches the required schema."
                )
            response = self.provider.extract(self._request(images, current_prompt, schema))
            try:
                payload = parse_json_text(response.text)
                return DocumentTypeDecision.model_validate(payload)
            except ValidationError as exc:
                last_error = InvalidResponseError(
                    "Document type validation failed: "
                    + "; ".join(validation_error_messages(exc)[:5])
                )
            except InvalidResponseError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _extract_envelope(
        self,
        images: tuple[VisionImage, ...],
        document_type: DocumentType,
        record: DocumentRecord,
        extracted_at: datetime,
    ) -> tuple[ExtractionEnvelope, DocumentPayload]:
        prompt = build_extraction_prompt(document_type, len(images))
        schema = ExtractionEnvelope.model_json_schema()
        validation_messages: list[str] = []
        last_details: list[ErrorDetail] = []

        for attempt in range(self.settings.max_retries + 1):
            current_prompt = (
                prompt
                if attempt == 0
                else build_correction_prompt(document_type, validation_messages)
            )
            response = self.provider.extract(self._request(images, current_prompt, schema))
            try:
                envelope = parse_extraction_envelope(response.text)
            except InvalidResponseError as exc:
                validation_messages = [str(exc)]
                last_details = [_issue("invalid_json", str(exc))]
                continue

            if envelope.document_type is not document_type:
                message = (
                    f"Model returned document_type {envelope.document_type.value!r} "
                    f"instead of the requested {document_type.value!r}."
                )
                validation_messages = [message]
                last_details = [_issue("document_type_mismatch", message, "document_type")]
                continue

            try:
                payload = parse_typed_document(
                    envelope.data,
                    document_type,
                    document_id=record.document_id,
                    source_filename=record.sanitized_filename,
                    extracted_at=extracted_at,
                    warnings=list(envelope.warnings),
                    missing_fields=list(envelope.missing_fields),
                    unreadable_fields=list(envelope.unreadable_fields),
                )
            except ValidationError as exc:
                validation_messages = validation_error_messages(exc)
                last_details = validation_error_details(exc)
                continue

            if payload is None:
                message = f"Unsupported document type: {document_type.value}."
                raise InvalidResponseError(message)
            return envelope, payload

        detail = last_details or [_issue("extraction_failed", "Extraction validation failed.")]
        raise _ValidationAttemptError(detail)

    def _confidence_map(
        self,
        envelope: ExtractionEnvelope,
        payload: DocumentPayload,
    ) -> tuple[dict[str, FieldConfidenceEntry], bool]:
        result: dict[str, FieldConfidenceEntry] = {}
        low_confidence = False
        for entry in envelope.field_confidence:
            confidence = entry.confidence
            unavailable = confidence is None
            below_threshold = (
                confidence is not None
                and confidence < self.settings.confidence_review_threshold
            )
            needs_review = entry.needs_review or unavailable or below_threshold
            if needs_review:
                low_confidence = True
            result[entry.field_path] = FieldConfidenceEntry(
                value=_lookup_value(payload, entry.field_path),
                confidence=confidence,
                evidence=entry.evidence,
                needs_review=needs_review,
            )
        return result, low_confidence

    def _base_result(
        self,
        record: DocumentRecord,
        document_type: DocumentType,
    ) -> ExtractionResult:
        return ExtractionResult(
            result_id=f"result-{new_document_id()}",
            document_type=document_type,
            source_document_id=record.document_id,
        )

    def extract_document(
        self,
        record: DocumentRecord,
        processed: PreprocessingResult,
        requested_type: DocumentType | str | None = None,
    ) -> ExtractionResult:
        """Extract one document using auto-detection or an explicit document type."""
        if isinstance(requested_type, str):
            try:
                requested_type = DocumentType(requested_type)
            except ValueError as exc:
                raise ValueError(f"Unsupported document type: {requested_type!r}.") from exc

        explicit_unknown = requested_type is DocumentType.UNKNOWN
        result = self._base_result(
            record,
            DocumentType.UNKNOWN if requested_type is None else requested_type,
        )
        images = _pages_to_images(processed)
        if not images:
            result.status = ProcessingStatus.FAILED
            result.validation_status = ValidationStatus.INVALID
            result.needs_review = True
            result.validation_errors = [
                _issue("no_prepared_pages", "No successfully prepared page images are available.")
            ]
            return result

        if explicit_unknown:
            result.status = ProcessingStatus.REQUIRES_REVIEW
            result.validation_status = ValidationStatus.NEEDS_REVIEW
            result.needs_review = True
            result.warnings.append("Unknown document type was selected; extraction was skipped.")
            return result

        try:
            if requested_type is None:
                decision = self._detect_document_type(images)
                if (
                    decision.document_type is DocumentType.UNKNOWN
                    or decision.confidence is None
                    or decision.confidence < self.settings.confidence_review_threshold
                ):
                    result.status = ProcessingStatus.REQUIRES_REVIEW
                    result.validation_status = ValidationStatus.NEEDS_REVIEW
                    result.needs_review = True
                    result.document_type = DocumentType.UNKNOWN
                    result.warnings.append(
                        "Document type could not be identified reliably; extraction was skipped."
                    )
                    if decision.reason:
                        result.warnings.append(f"Detection note: {decision.reason[:500]}")
                    return result
                result.document_type = decision.document_type
                document_type = decision.document_type
            else:
                document_type = requested_type

            extracted_at = _utc_now()
            envelope, payload = self._extract_envelope(
                images, document_type, record, extracted_at
            )
        except _ValidationAttemptError as exc:
            result.status = ProcessingStatus.FAILED
            result.validation_status = ValidationStatus.INVALID
            result.needs_review = True
            result.validation_errors = exc.details
            result.warnings.append("Schema validation failed after bounded retries.")
            return result
        except ProviderError as exc:
            result.status = ProcessingStatus.FAILED
            result.validation_status = ValidationStatus.INVALID
            result.needs_review = True
            result.validation_errors = [
                _issue(exc.code, exc.user_message),
            ]
            return result
        except InvalidResponseError as exc:
            result.status = ProcessingStatus.FAILED
            result.validation_status = ValidationStatus.INVALID
            result.needs_review = True
            result.validation_errors = [_issue("invalid_response", str(exc))]
            return result

        field_confidence, low_confidence = self._confidence_map(envelope, payload)
        warnings = list(envelope.warnings)
        warnings.append(_CONFIDENCE_NOTE)
        if envelope.missing_fields:
            warnings.append("Missing fields: " + ", ".join(envelope.missing_fields[:20]))
        if envelope.unreadable_fields:
            warnings.append(
                "Unreadable fields: " + ", ".join(envelope.unreadable_fields[:20])
            )
        if envelope.confidence_note:
            warnings.append(f"Confidence note: {envelope.confidence_note[:500]}")
        warnings.extend(_arithmetic_warnings(payload))
        if low_confidence:
            warnings.append("One or more fields are low-confidence or unreadable.")

        needs_review = bool(
            low_confidence
            or envelope.missing_fields
            or envelope.unreadable_fields
            or any("inconsistent" in warning or "do not match" in warning for warning in warnings)
        )
        result.data = payload
        result.status = (
            ProcessingStatus.REQUIRES_REVIEW if needs_review else ProcessingStatus.COMPLETED
        )
        result.validation_status = (
            ValidationStatus.NEEDS_REVIEW
            if needs_review
            else ValidationStatus.VALID_WITH_WARNINGS
            if warnings
            else ValidationStatus.VALID
        )
        result.validation_errors = []
        result.warnings = warnings
        result.field_confidence = field_confidence
        result.overall_confidence = envelope.overall_confidence
        result.needs_review = needs_review
        result.processed_at = extracted_at
        return result


class _ValidationAttemptError(Exception):
    def __init__(self, details: list[ErrorDetail]) -> None:
        self.details = details
        super().__init__("Extraction validation failed after retries.")
