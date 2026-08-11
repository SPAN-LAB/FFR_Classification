"""Run one subject/model/analysis-value unit and write its result files."""

from __future__ import annotations

import argparse
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.job_result import write_result_files
from src.constants.defaults import (
    NUM_FOLDS,
    SUBAVERAGE_SIZE,
    TRIM_END_TIME,
    TRIM_START_TIME,
)
from src.core import AnalysisPipeline

from .training_options import (
    ANALYSIS_TRAINING_OPTIONS,
    DEFAULT_ANALYSIS_TRAINING_OPTIONS,
)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one FFR analysis value for one subject and model."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--subject", required=True, help="Path to one subject .mat file")
    parser.add_argument(
        "--analysis",
        required=True,
        choices=["subaverage", "data_amount"],
    )
    parser.add_argument(
        "--value",
        required=True,
        type=int,
        help="Subaverage size or training examples per fold",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Path prefix for .summary.json and .predictions.csv outputs",
    )
    parser.add_argument("--cluster-id", default=None)
    parser.add_argument("--process-id", default=None)
    return parser


def _run_analysis(args: argparse.Namespace) -> AnalysisPipeline:
    options = ANALYSIS_TRAINING_OPTIONS.get(
        args.model,
        DEFAULT_ANALYSIS_TRAINING_OPTIONS,
    )
    pipeline = (
        AnalysisPipeline()
        .load_subjects(args.subject)
        .trim_by_timestamp(
            start_time=TRIM_START_TIME,
            end_time=TRIM_END_TIME,
        )
    )

    if args.analysis == "subaverage":
        return (
            pipeline
            .subaverage(size=args.value)
            .fold(num_folds=NUM_FOLDS)
            .evaluate_model(
                model_name=args.model,
                training_options=options,
            )
        )

    return (
        pipeline
        .subaverage(size=SUBAVERAGE_SIZE)
        .fold(num_folds=NUM_FOLDS)
        .evaluate_model_with_training_amount(
            model_name=args.model,
            training_options=options,
            training_amount=args.value,
        )
    )


def main() -> None:
    args = _parser().parse_args()
    if args.value < 1:
        raise ValueError("--value must be at least 1")

    output_prefix = args.output_prefix
    if output_prefix is None:
        subject = Path(args.subject).stem
        output_prefix = (
            Path("analyses")
            / args.analysis
            / f"{args.model}.{subject}.{args.analysis}-{args.value}"
        )

    started_at = datetime.now(timezone.utc)
    started_timer = time.monotonic()
    pipeline = None
    error = None
    status = "success"

    try:
        pipeline = _run_analysis(args)
    except Exception:
        status = "failure"
        error = traceback.format_exc()

    finished_at = datetime.now(timezone.utc)
    metadata = {
        "status": status,
        "subject": Path(args.subject).stem,
        "subject_filepath": args.subject,
        "model": args.model,
        "analysis": args.analysis,
        "value": args.value,
        "value_definition": (
            "waveforms averaged per example"
            if args.analysis == "subaverage"
            else "training examples per cross-validation fold"
        ),
        "training_options": ANALYSIS_TRAINING_OPTIONS.get(
            args.model,
            DEFAULT_ANALYSIS_TRAINING_OPTIONS,
        ),
        "git_commit": _git_commit(),
        "condor_cluster_id": args.cluster_id,
        "condor_process_id": args.process_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": time.monotonic() - started_timer,
    }
    summary_path, predictions_path = write_result_files(
        output_prefix,
        pipeline=pipeline,
        metadata=metadata,
        error_traceback=error,
    )
    print(f"[run_job] Summary written to {summary_path.resolve()}")
    print(f"[run_job] Predictions written to {predictions_path.resolve()}")

    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
