"""Tests for the local Phase 5 evaluation framework."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.evaluation.cli import _resolve_reports_root, main as cli_main
from src.evaluation.comparison import compare_values, flatten_payload, normalize_string
from src.evaluation.dataset import (
    DEFAULT_EVALUATION_ROOT,
    load_dataset,
    load_ground_truths,
    select_samples,
    validate_dataset,
)
from src.evaluation.fixtures import FixtureVisionProvider, unflatten_fields
from src.evaluation.metrics import compute_field_metrics
from src.evaluation.reporting import render_markdown_report
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schemas import (
    ComparisonStatus,
    DocumentEvaluation,
    EvaluationMode,
    EvaluationSubset,
    FieldComparison,
    GroundTruth,
)
from src.schemas.common import DocumentType
from src.schemas.diagram import DiagramType

from .conftest import make_extraction_settings


def test_dataset_is_valid() -> None:
    validation = validate_dataset(DEFAULT_EVALUATION_ROOT)
    assert validation["valid"] is True
    assert validation["errors"] == []
    assert validation["sample_count"] == 11
    assert validation["subset_counts"] == {"full": 7, "small": 4}


def test_ground_truths_match_sample_files() -> None:
    samples = load_dataset(DEFAULT_EVALUATION_ROOT)
    assert len(samples) == 11
    assert all(sample.file_path.is_file() for sample in samples)
    assert len({sample.sample_id for sample in samples}) == 11


def test_load_dataset_rejects_sample_path_outside_samples_root(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    (root / "ground_truth").mkdir(parents=True)
    (root / "samples").mkdir()
    (tmp_path / "outside.txt").write_text("not a sample", encoding="utf-8")
    truth = GroundTruth(
        sample_id="sample-path-escape",
        ground_truth_id="gt-path-escape",
        document_type=DocumentType.INVOICE,
        expected_document_type=DocumentType.INVOICE,
        file_name="../outside.txt",
        dataset_version="phase5-synthetic-v1",
        source="test fixture",
        license="CC0-1.0",
        checked_by="test",
        checked_at=date(2026, 1, 1),
        expected_fields={"vendor_name": "Acme"},
    )
    (root / "ground_truth" / "sample.json").write_text(
        truth.model_dump_json(), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="outside the samples directory"):
        load_dataset(root)


def test_comparison_normalizes_enum_decimal_and_whitespace() -> None:
    assert compare_values("flowchart", DiagramType.FLOWCHART).normalized_match is True
    assert compare_values("10.00", "10.00", numeric_tolerance=Decimal("0.01")).normalized_match is True
    assert normalize_string("  Acme   Test Supply ") == "acme test supply"


def test_unflatten_fields_handles_nested_lists() -> None:
    result = unflatten_fields(
        {
            "vendor_name": "Acme",
            "line_items[0].description": "Widget",
            "line_items[1].description": "Tool",
        }
    )
    assert result == {
        "vendor_name": "Acme",
        "line_items": [
            {"description": "Widget"},
            {"description": "Tool"},
        ],
    }


def test_flatten_payload_skips_document_metadata() -> None:
    flattened = flatten_payload(
        {
            "document_id": "doc-1",
            "vendor_name": "Acme",
            "line_items": [{"description": "Widget"}],
        }
    )
    assert "document_id" not in flattened
    assert flattened["vendor_name"] == "Acme"
    assert flattened["line_items[0].description"] == "Widget"


def test_field_metrics_counts_statuses() -> None:
    evaluation = DocumentEvaluation(
        run_id="run-test",
        sample_id="sample-test",
        document_type="invoice",
        expected_document_type="invoice",
        extraction_status="completed",
        schema_valid=True,
        mode=EvaluationMode.LOCAL_SYNTHETIC_TESTS,
        field_metrics={},
        comparisons=(
            FieldComparison(
                field_path="vendor_name",
                expected_value="Acme",
                extracted_value="Acme",
                status=ComparisonStatus.CORRECT,
                exact_match=True,
                normalized_match=True,
                comparison_method="string",
            ),
            FieldComparison(
                field_path="total",
                expected_value="11.00",
                extracted_value=None,
                status=ComparisonStatus.MISSING,
                exact_match=False,
                normalized_match=False,
                comparison_method="presence",
            ),
        ),
        latency_seconds=0.01,
        provider_requests=2,
        retry_count=0,
        fingerprint="fingerprint",
    )
    metrics = compute_field_metrics([evaluation])
    assert metrics["compared_fields"] == 2
    assert metrics["correct"] == 1
    assert metrics["missing"] == 1
    assert metrics["normalized_accuracy"] == 0.5


def test_local_runner_uses_fixture_without_network(tmp_path: Path) -> None:
    settings: Settings = make_extraction_settings(EVALUATION_MAX_WORKERS="1")
    samples = select_samples(
        load_dataset(DEFAULT_EVALUATION_ROOT),
        EvaluationSubset.SMALL,
    )
    provider = FixtureVisionProvider(
        [sample.ground_truth for sample in samples],
        name="test_fixture",
    )
    runner = EvaluationRunner(
        settings,
        EvaluationMode.LOCAL_SYNTHETIC_TESTS,
        results_root=tmp_path,
        provider=provider,
        max_workers=1,
        request_delay_seconds=0,
    )
    summary = runner.run(subset=EvaluationSubset.SMALL, resume=False)

    assert summary.api_status == "local_only"
    assert summary.sample_count == 4
    assert summary.schema_validity["valid_count"] == 4
    assert summary.field_metrics["normalized_accuracy"] == 1.0
    assert summary.document_type_metrics["document_type_accuracy"] == 1.0
    assert len(list(tmp_path.rglob("*.json"))) >= 5
    assert "no live Gemini API calls were made" in render_markdown_report(summary)


def test_cli_reports_root_defaults_to_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    assert _resolve_reports_root(
        SimpleNamespace(dataset_root=dataset_root, reports_root=None)
    ) == dataset_root / "reports"
    explicit = tmp_path / "custom-reports"
    assert _resolve_reports_root(
        SimpleNamespace(dataset_root=dataset_root, reports_root=explicit)
    ) == explicit


def test_cli_does_not_print_raw_exception_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "raw-exception-secret"

    class FailingRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(f"boom {secret}")

    monkeypatch.setattr("src.evaluation.cli.load_settings", make_extraction_settings)
    monkeypatch.setattr("src.evaluation.cli.EvaluationRunner", FailingRunner)
    assert cli_main(["run", "--mode", "local", "--subset", "small"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "RuntimeError" in captured.err


def test_live_cli_requires_explicit_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.evaluation.cli.load_settings", make_extraction_settings)
    assert cli_main(["run", "--mode", "live", "--subset", "small"]) == 2
    assert (
        cli_main(
            [
                "run",
                "--mode",
                "live",
                "--subset",
                "full",
                "--confirm-live",
            ]
        )
        == 2
    )


def test_report_json_is_serializable(tmp_path: Path) -> None:
    settings: Settings = make_extraction_settings()
    samples = select_samples(load_dataset(DEFAULT_EVALUATION_ROOT), EvaluationSubset.SMALL)
    runner = EvaluationRunner(
        settings,
        EvaluationMode.MOCKED_API_TESTS,
        results_root=tmp_path,
        provider=FixtureVisionProvider([sample.ground_truth for sample in samples]),
        max_workers=1,
        request_delay_seconds=0,
    )
    summary = runner.run(subset=EvaluationSubset.SMALL, resume=False)
    payload = json.loads(json.dumps(summary.model_dump(mode="json")))
    assert payload["mode"] == "mocked_api_tests"
    assert payload["api_status"] == "mocked_provider"
    assert len(payload["document_evaluations"]) == 4
