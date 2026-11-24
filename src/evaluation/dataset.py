"""Loading and validation for the Phase 5 synthetic evaluation dataset."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.config import PROJECT_ROOT

from .schemas import EvaluationSubset, GroundTruth

DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "evaluation"


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    file_path: Path
    ground_truth: GroundTruth
    sha256: str
    size_bytes: int

    @property
    def subset(self) -> EvaluationSubset:
        return self.ground_truth.dataset_subset


def load_index(root: Path | None = None) -> dict[str, Any]:
    evaluation_root = root or DEFAULT_EVALUATION_ROOT
    path = evaluation_root / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation index is missing: {path}")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation index must be a JSON object.")
    return payload


def sample_version(root: Path | None = None) -> str:
    try:
        return str(load_index(root).get("dataset_version", "unknown"))
    except (FileNotFoundError, ValueError, OSError):
        return "unknown"


def load_ground_truths(root: Path | None = None) -> list[GroundTruth]:
    evaluation_root = root or DEFAULT_EVALUATION_ROOT
    truth_dir = evaluation_root / "ground_truth"
    if not truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory is missing: {truth_dir}")
    truths: list[GroundTruth] = []
    for path in sorted(truth_dir.glob("*.json")):
        truth = GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
        truths.append(truth)
    return truths


def _resolve_sample_path(evaluation_root: Path, truth: GroundTruth) -> Path:
    samples_root = (evaluation_root / "samples").resolve()
    candidate = (samples_root / truth.file_name).resolve()
    try:
        candidate.relative_to(samples_root)
    except ValueError as exc:
        raise ValueError(
            "Sample file path is outside the samples directory for "
            f"{truth.sample_id}: {truth.file_name!r}."
        ) from exc
    return candidate


def load_dataset(root: Path | None = None) -> list[EvaluationSample]:
    evaluation_root = root or DEFAULT_EVALUATION_ROOT
    truths = load_ground_truths(evaluation_root)
    samples: list[EvaluationSample] = []
    for truth in truths:
        file_path = _resolve_sample_path(evaluation_root, truth)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Sample file is missing for {truth.sample_id}: {truth.file_name!r}"
            )
        data = file_path.read_bytes()
        samples.append(
            EvaluationSample(
                sample_id=truth.sample_id,
                file_path=file_path,
                ground_truth=truth,
                sha256=sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    return samples


def select_samples(
    samples: list[EvaluationSample],
    subset: EvaluationSubset | str = EvaluationSubset.FULL,
    sample_ids: list[str] | None = None,
) -> list[EvaluationSample]:
    requested = EvaluationSubset(subset)
    selected = [
        sample
        for sample in samples
        if requested is EvaluationSubset.FULL or sample.subset is requested
    ]
    if sample_ids is not None:
        id_set = set(sample_ids)
        selected = [sample for sample in selected if sample.sample_id in id_set]
    return selected


def validate_dataset(
    root: Path | None = None,
    samples: list[EvaluationSample] | None = None,
) -> dict[str, Any]:
    evaluation_root = root or DEFAULT_EVALUATION_ROOT
    errors: list[str] = []
    warnings: list[str] = []
    try:
        index = load_index(evaluation_root)
    except (FileNotFoundError, ValueError, OSError, ValidationError) as exc:
        index = {}
        errors.append(f"index: {type(exc).__name__}: {exc}")

    try:
        loaded = samples if samples is not None else load_dataset(evaluation_root)
    except (FileNotFoundError, ValueError, OSError, ValidationError) as exc:
        loaded = []
        errors.append(f"dataset: {type(exc).__name__}: {exc}")

    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for sample in loaded:
        truth = sample.ground_truth
        if truth.sample_id in seen_ids:
            errors.append(f"{truth.sample_id}: duplicate sample_id.")
        seen_ids.add(truth.sample_id)
        if truth.file_name in seen_files:
            errors.append(f"{truth.sample_id}: duplicate file_name {truth.file_name!r}.")
        seen_files.add(truth.file_name)
        if not truth.expected_fields:
            errors.append(f"{truth.sample_id}: expected_fields is empty.")
        if truth.schema_version != "5.0":
            errors.append(f"{truth.sample_id}: unsupported schema_version {truth.schema_version!r}.")
        if index and truth.dataset_version != index.get("dataset_version"):
            errors.append(
                f"{truth.sample_id}: dataset_version {truth.dataset_version!r} "
                f"does not match index {index.get('dataset_version')!r}."
            )
        if truth.expected_document_type is not truth.document_type:
            errors.append(f"{truth.sample_id}: document type expectations disagree.")
        if not truth.source or not truth.license:
            errors.append(f"{truth.sample_id}: source or license is missing.")

    manifest_path = evaluation_root / "samples" / "manifest.json"
    if manifest_path.is_file():
        import json

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes = {
                item["file_name"]: item["sha256"]
                for item in manifest.get("files", [])
                if isinstance(item, dict) and "file_name" in item
            }
            for sample in loaded:
                expected_hash = hashes.get(sample.ground_truth.file_name)
                if expected_hash and expected_hash != sample.sha256:
                    errors.append(f"{sample.sample_id}: sample hash differs from manifest.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"manifest could not be checked: {type(exc).__name__}: {exc}")
    else:
        warnings.append("samples/manifest.json is missing; hash check was skipped.")

    type_counts: dict[str, int] = {}
    subset_counts: dict[str, int] = {}
    for sample in loaded:
        type_name = sample.ground_truth.document_type.value
        subset_name = sample.subset.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        subset_counts[subset_name] = subset_counts.get(subset_name, 0) + 1

    if type_counts.get("invoice", 0) < 2:
        warnings.append("Fewer than two invoice samples; per-type findings are limited.")
    if type_counts.get("receipt", 0) < 2:
        warnings.append("Fewer than two receipt samples; per-type findings are limited.")
    if not any("multi-page" in sample.ground_truth.tags for sample in loaded):
        warnings.append("No multi-page sample is tagged.")

    return {
        "dataset_version": index.get("dataset_version", "unknown"),
        "sample_count": len(loaded),
        "document_type_counts": dict(sorted(type_counts.items())),
        "subset_counts": dict(sorted(subset_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "document_type": sample.ground_truth.document_type.value,
                "file_name": sample.ground_truth.file_name,
                "subset": sample.subset.value,
                "tags": list(sample.ground_truth.tags),
                "sha256": sample.sha256,
                "size_bytes": sample.size_bytes,
            }
            for sample in loaded
        ],
    }
