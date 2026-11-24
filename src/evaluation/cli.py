"""Command-line interface for Phase 5 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.config import load_settings
from src.evaluation.dataset import (
    DEFAULT_EVALUATION_ROOT,
    load_dataset,
    select_samples,
    validate_dataset,
)
from src.evaluation.fixtures import FixtureVisionProvider
from src.evaluation.reporting import write_reports
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schemas import EvaluationMode, EvaluationSubset

_MODE_MAP = {
    "local": EvaluationMode.LOCAL_SYNTHETIC_TESTS,
    "mock": EvaluationMode.MOCKED_API_TESTS,
    "live": EvaluationMode.REAL_GEMINI_API_EVALUATION,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase5-eval",
        description="Run and validate repeatable document-intelligence evaluations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate the synthetic dataset.")
    validate.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
        help="Evaluation dataset root directory.",
    )

    run = subparsers.add_parser("run", help="Run an evaluation and write reports.")
    run.add_argument("--mode", choices=sorted(_MODE_MAP), required=True)
    run.add_argument("--subset", choices=[item.value for item in EvaluationSubset], default="small")
    run.add_argument("--dataset-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    run.add_argument("--reports-root", type=Path, default=None)
    run.add_argument("--sample-id", action="append", default=None)
    run.add_argument("--max-workers", type=int, default=None)
    run.add_argument("--request-delay", type=float, default=None)
    run.add_argument(
        "--formats",
        nargs="+",
        choices=("json", "markdown", "csv"),
        default=("json", "markdown", "csv"),
    )
    run.add_argument("--resume", action="store_true", help="Resume only for live evaluation.")
    run.add_argument("--no-resume", action="store_true", help="Disable resume for live evaluation.")
    run.add_argument(
        "--confirm-live",
        action="store_true",
        help="Explicit consent to make real, billable Gemini API requests.",
    )
    run.add_argument(
        "--confirm-full",
        action="store_true",
        help="Explicit consent to run the full subset after a small live run.",
    )
    return parser


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _resolve_reports_root(args: argparse.Namespace) -> Path:
    return args.reports_root or (args.dataset_root / "reports")


def _validate_command(args: argparse.Namespace) -> int:
    _print_json(validate_dataset(args.dataset_root))
    return 0


def _run_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    mode = _MODE_MAP[args.mode]
    subset = EvaluationSubset(args.subset)
    reports_root = _resolve_reports_root(args)

    if mode is EvaluationMode.REAL_GEMINI_API_EVALUATION:
        if not args.confirm_live:
            print(
                "Refusing to call the live API without --confirm-live. "
                "This mode may incur cost and consume quota.",
                file=sys.stderr,
            )
            return 2
        if not settings.is_api_key_configured:
            print(
                "GEMINI_API_KEY is not configured. Add it to .env and rerun with --confirm-live.",
                file=sys.stderr,
            )
            return 2
        if subset is EvaluationSubset.FULL and not args.confirm_full:
            print(
                "Refusing the full live subset without --confirm-full. "
                "Run --subset small first.",
                file=sys.stderr,
            )
            return 2
        resume = not args.no_resume
        provider = None
        max_workers = args.max_workers
        request_delay = args.request_delay
    else:
        if args.confirm_live or args.confirm_full:
            print("Live confirmation flags are only valid with --mode live.", file=sys.stderr)
            return 2
        if args.resume:
            print("Resume is available only for live evaluation.", file=sys.stderr)
            return 2
        selected = select_samples(load_dataset(args.dataset_root), subset, args.sample_id)
        provider = FixtureVisionProvider(
            [sample.ground_truth for sample in selected],
            name="local_fixture" if mode is EvaluationMode.LOCAL_SYNTHETIC_TESTS else "mock_fixture",
        )
        resume = False
        max_workers = 1
        request_delay = 0.0

    runner = EvaluationRunner(
        settings,
        mode,
        dataset_root=args.dataset_root,
        provider=provider,
        max_workers=max_workers,
        request_delay_seconds=request_delay,
    )

    def progress(index: int, total: int, sample_id: str) -> None:
        print(f"[{index}/{total}] {sample_id}", file=sys.stderr)

    try:
        summary = runner.run(
            subset=subset,
            resume=resume,
            sample_ids=args.sample_id,
            progress_callback=progress,
        )
        outputs = write_reports(summary, reports_root, formats=args.formats)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}.", file=sys.stderr)
        return 1

    _print_json(
        {
            "run_id": summary.run_id,
            "mode": summary.mode.value,
            "subset": summary.subset.value,
            "api_status": summary.api_status,
            "sample_count": summary.sample_count,
            "schema_validity_rate": summary.schema_validity.get("schema_validity_rate"),
            "normalized_accuracy": summary.field_metrics.get("normalized_accuracy"),
            "documents": len(summary.document_evaluations),
            "reports": outputs,
            "raw_results_directory": summary.raw_results_directory,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate_command(args)
    if args.command == "run":
        return _run_command(args)
    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
