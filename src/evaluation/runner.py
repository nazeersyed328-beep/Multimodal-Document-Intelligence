"""Repeatable local and live evaluation runner."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from src.config import PROJECT_ROOT, Settings
from src.document.ingestion import ingest_bytes
from src.document.preprocessing import preprocess_document
from src.extraction import ExtractionService
from src.extraction.export import extraction_result_to_dict
from src.extraction.provider import (
    ProviderError,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    create_vision_provider,
)
from src.quality import QualityAssessmentService
from src.quality.export import quality_report_to_dict
from src.schemas.common import ErrorDetail, ErrorSeverity, ValidationStatus
from src.schemas.extraction_result import ExtractionResult, ProcessingStatus

from .comparison import evaluate_fields, jsonable
from .dataset import (
    DEFAULT_EVALUATION_ROOT,
    EvaluationSample,
    load_dataset,
    sample_version,
    select_samples,
    validate_dataset,
)
from .metrics import (
    build_error_analysis,
    compute_document_type_metrics,
    compute_field_metrics,
    compute_field_metrics_from_comparisons,
    compute_per_document_type_metrics,
    compute_performance,
    compute_quality_metrics,
    compute_schema_validity,
)
from .schemas import (
    DocumentEvaluation,
    EvaluationMode,
    EvaluationRunSummary,
    EvaluationSubset,
)

ProgressCallback = Callable[[int, int, str], None]


class CountingProvider(VisionProvider):
    """Wrap a provider to count requests and apply optional rate limiting."""

    def __init__(self, delegate: VisionProvider, delay_seconds: float = 0.0) -> None:
        self.delegate = delegate
        self.name = getattr(delegate, "name", "counted")
        self.delay_seconds = max(0.0, delay_seconds)
        self.request_count = 0
        self.detection_requests = 0
        self.extraction_requests = 0
        self.error_count = 0

    def extract(self, request: VisionRequest) -> VisionResponse:
        self.request_count += 1
        schema_title = str(request.response_schema.get("title", ""))
        if schema_title == "DocumentTypeDecision":
            self.detection_requests += 1
        elif schema_title == "ExtractionEnvelope":
            self.extraction_requests += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        try:
            return self.delegate.extract(request)
        except Exception:
            self.error_count += 1
            raise


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class EvaluationRunner:
    """Run repeatable evaluations with resume and saved raw outputs."""

    def __init__(
        self,
        settings: Settings,
        mode: EvaluationMode,
        *,
        dataset_root: Path | None = None,
        results_root: Path | None = None,
        provider: VisionProvider | None = None,
        max_workers: int | None = None,
        request_delay_seconds: float | None = None,
        numeric_tolerance: Decimal | None = None,
    ) -> None:
        self.settings = settings
        self.mode = EvaluationMode(mode)
        self.dataset_root = dataset_root or DEFAULT_EVALUATION_ROOT
        self.results_root = results_root or (self.dataset_root / "results")
        self.numeric_tolerance = numeric_tolerance or settings.amount_tolerance
        self.max_workers = max(1, max_workers or settings.evaluation_max_workers)
        self.request_delay_seconds = (
            settings.evaluation_request_delay_seconds
            if request_delay_seconds is None
            else max(0.0, request_delay_seconds)
        )
        self._provider = provider
        if self._provider is None:
            if self.mode is not EvaluationMode.REAL_GEMINI_API_EVALUATION:
                raise ValueError("Offline evaluation modes require an injected provider.")
            if not settings.is_api_key_configured:
                raise ValueError("GEMINI_API_KEY is not configured for live evaluation.")
            self._provider = create_vision_provider(settings)

    def _fingerprint(self, sample: EvaluationSample) -> str:
        payload = {
            "sample_id": sample.sample_id,
            "sample_sha256": sample.sha256,
            "ground_truth": jsonable(sample.ground_truth.model_dump(mode="python")),
            "mode": self.mode.value,
            "model_id": self.settings.gemini_model,
            "dataset_version": sample.ground_truth.dataset_version,
            "confidence_threshold": self.settings.confidence_review_threshold,
            "numeric_tolerance": str(self.numeric_tolerance),
            "max_quality_issues": self.settings.max_quality_issues,
            "future_date_tolerance_days": self.settings.future_date_tolerance_days,
            "max_past_date_days": self.settings.max_past_date_days,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _run_paths(self, subset: EvaluationSubset) -> tuple[str, Path]:
        key = _safe_name(
            f"{self.mode.value}_{subset.value}_{self.settings.gemini_model}_"
            f"{sample_version(self.dataset_root)}"
        )
        directory = self.results_root / key
        return key, directory

    def _failed_result(
        self,
        sample: EvaluationSample,
        document_id: str,
        code: str,
        message: str,
    ) -> ExtractionResult:
        return ExtractionResult(
            result_id=f"result-{sample.sample_id}",
            document_type=sample.ground_truth.expected_document_type,
            source_document_id=document_id,
            status=ProcessingStatus.FAILED,
            validation_status=ValidationStatus.INVALID,
            validation_errors=[
                ErrorDetail(
                    code=code,
                    message=message[:2000],
                    severity=ErrorSeverity.ERROR,
                )
            ],
            needs_review=True,
            processed_at=datetime.now(timezone.utc),
        )

    def _error_info(self, exc: Exception) -> tuple[str, str]:
        if isinstance(exc, ProviderError):
            return exc.code, exc.user_message
        return type(exc).__name__, f"{type(exc).__name__} during evaluation."

    def _evaluate_one(
        self,
        sample: EvaluationSample,
        run_id: str,
        raw_directory: Path,
    ) -> DocumentEvaluation:
        fingerprint = self._fingerprint(sample)
        raw_path = raw_directory / f"{_safe_name(sample.sample_id)}.json"

        started = time.perf_counter()
        counting = CountingProvider(self._provider, self.request_delay_seconds)
        document_id = f"eval-{_safe_name(sample.sample_id)}"
        error_code: str | None = None
        error_message: str | None = None
        result: ExtractionResult | None = None
        processed = None
        try:
            data = sample.file_path.read_bytes()
            ingestion, record = ingest_bytes(
                sample.ground_truth.file_name,
                data,
                self.settings,
                upload_id=document_id,
            )
            if record is None:
                raise ValueError(f"Ingestion rejected sample: {sample.sample_id}")
            document_id = record.document_id
            processed = preprocess_document(record, self.settings)
            service = ExtractionService(self.settings, provider=counting)
            result = service.extract_document(record, processed, requested_type=None)
            if result.status is ProcessingStatus.FAILED and result.validation_errors:
                error_code = result.validation_errors[0].code
                error_message = result.validation_errors[0].message
        except Exception as exc:
            error_code, error_message = self._error_info(exc)
            result = self._failed_result(sample, document_id, error_code, error_message)

        quality = QualityAssessmentService(self.settings).assess(
            result,
            preprocessing=processed,
            detected_type=None,
        )
        comparisons = evaluate_fields(
            sample.ground_truth,
            result,
            confidence_threshold=self.settings.confidence_review_threshold,
            numeric_tolerance=self.numeric_tolerance,
        )
        field_metrics = compute_field_metrics_from_comparisons(comparisons)
        schema_valid = bool(
            result.data is not None and result.validation_status is not ValidationStatus.INVALID
        )
        unreadable = list(getattr(result.data, "unreadable_fields", ()) or ())
        unreadable.extend(quality.validation_summary.unreadable_fields)
        low_confidence = [
            review.field_path
            for review in quality.field_review
            if review.confidence is not None
            and review.confidence < self.settings.confidence_review_threshold
        ]
        latency = round(time.perf_counter() - started, 6)
        evaluation = DocumentEvaluation(
            run_id=run_id,
            sample_id=sample.sample_id,
            document_type=result.document_type.value,
            expected_document_type=sample.ground_truth.expected_document_type.value,
            extraction_status=result.status.value,
            schema_valid=schema_valid,
            mode=self.mode,
            field_metrics=field_metrics,
            comparisons=tuple(comparisons),
            detected_issue_rules=tuple(sorted({issue.rule_id for issue in quality.issues})),
            expected_issue_rules=sample.ground_truth.expected_issue_rules,
            flagged_low_confidence_fields=tuple(sorted(set(low_confidence))),
            flagged_unreadable_fields=tuple(sorted(set(unreadable))),
            quality_review_required=quality.review_required,
            expected_review_required=sample.ground_truth.expected_review_required,
            latency_seconds=latency,
            provider_requests=counting.request_count,
            retry_count=max(0, counting.detection_requests - 1)
            + max(0, counting.extraction_requests - 1),
            error_code=error_code,
            error_message=error_message,
            fingerprint=fingerprint,
        )
        raw_payload = {
            "fingerprint": fingerprint,
            "sample_id": sample.sample_id,
            "mode": self.mode.value,
            "model_id": self.settings.gemini_model,
            "extraction": extraction_result_to_dict(result),
            "quality": quality_report_to_dict(quality),
            "evaluation": evaluation.model_dump(mode="json"),
        }
        raw_directory.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(raw_payload, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        return evaluation

    def run(
        self,
        *,
        subset: EvaluationSubset | str = EvaluationSubset.SMALL,
        resume: bool = True,
        sample_ids: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> EvaluationRunSummary:
        requested_subset = EvaluationSubset(subset)
        dataset_validation = validate_dataset(self.dataset_root)
        if not dataset_validation["valid"]:
            raise ValueError(
                "Evaluation dataset is invalid: " + "; ".join(dataset_validation["errors"][:5])
            )
        samples = select_samples(load_dataset(self.dataset_root), requested_subset, sample_ids)
        if not samples:
            raise ValueError("No evaluation samples matched the requested subset.")

        _, raw_directory = self._run_paths(requested_subset)
        timestamp = datetime.now(timezone.utc)
        run_id = f"{_safe_name(self.mode.value)}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        evaluations: list[DocumentEvaluation] = []
        resumed = 0
        total = len(samples)

        def handle(sample: EvaluationSample) -> DocumentEvaluation:
            nonlocal resumed
            existing_path = raw_directory / f"{_safe_name(sample.sample_id)}.json"
            if resume and existing_path.is_file():
                try:
                    saved = json.loads(existing_path.read_text(encoding="utf-8"))
                    if saved.get("fingerprint") == self._fingerprint(sample):
                        resumed += 1
                        return DocumentEvaluation.model_validate(saved["evaluation"])
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            return self._evaluate_one(sample, run_id, raw_directory)

        if self.max_workers == 1:
            for index, sample in enumerate(samples, start=1):
                evaluations.append(handle(sample))
                if progress_callback:
                    progress_callback(index, total, sample.sample_id)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(handle, sample) for sample in samples]
                for index, future in enumerate(futures, start=1):
                    evaluations.append(future.result())
                    if progress_callback:
                        progress_callback(index, total, samples[index - 1].sample_id)

        truths = {sample.sample_id: sample.ground_truth for sample in samples}
        field_metrics = compute_field_metrics(evaluations)
        quality_metrics = compute_quality_metrics(evaluations, truths)
        performance = compute_performance(evaluations)
        recommendations: list[str] = []
        if compute_schema_validity(evaluations)["invalid_count"]:
            recommendations.append("Inspect schema failures before expanding the dataset.")
        if field_metrics.get("missing", 0):
            recommendations.append("Review missing-field failures and source rendering quality.")
        if (compute_document_type_metrics(evaluations) or {}).get("incorrect", 0):
            recommendations.append("Review document-type prompts for misclassified samples.")
        recommendations.append("Run the small live subset before the full dataset.")

        summary = EvaluationRunSummary(
            run_id=run_id,
            generated_at=timestamp,
            mode=self.mode,
            subset=requested_subset,
            dataset_version=sample_version(self.dataset_root),
            model_id=self.settings.gemini_model,
            api_status=(
                "live_api_configured"
                if self.mode is EvaluationMode.REAL_GEMINI_API_EVALUATION
                else ("mocked_provider" if self.mode is EvaluationMode.MOCKED_API_TESTS else "local_only")
            ),
            sample_count=len(evaluations),
            schema_validity=compute_schema_validity(evaluations),
            field_metrics=field_metrics,
            document_type_metrics=compute_document_type_metrics(evaluations),
            per_document_type_metrics=compute_per_document_type_metrics(evaluations),
            quality_metrics=quality_metrics,
            performance=performance,
            document_evaluations=tuple(evaluations),
            error_analysis=tuple(build_error_analysis(evaluations, truths)),
            limitations=(
                "Synthetic samples do not represent all real-world layouts or handwriting styles.",
                "Confidence values are model estimates, not calibrated probabilities.",
                "Small supports make some precision, recall, and F1 values explanatory only.",
                "Retry count is inferred from repeated detection or extraction requests beyond the first of each type.",
                "Quality rule expectations are manually authored for the synthetic dataset.",
            ),
            recommendations=tuple(recommendations),
            raw_results_directory=_display_path(raw_directory),
            resumed_count=resumed,
            dataset_validation=dataset_validation,
        )
        raw_directory.mkdir(parents=True, exist_ok=True)
        (raw_directory / "summary.json").write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return summary
