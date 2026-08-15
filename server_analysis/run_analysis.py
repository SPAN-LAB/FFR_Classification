"""Run one complete FFR analysis sweep or selected analysis values."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.job_result import write_analysis_results
from src.analysis.runner import run_analysis_conditions
from src.analysis.settings import ANALYSIS_TYPES, training_options_for


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
        description=(
            "Run an FFR analysis. Omit --value to run every configured value for "
            "the selected analysis."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--subject", required=True, help="Path to one subject .mat file")
    parser.add_argument("--analysis", required=True, choices=ANALYSIS_TYPES)
    parser.add_argument(
        "--value",
        action="append",
        type=int,
        help="Run only this value. May be supplied more than once.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Path prefix for the summary JSON and predictions CSV",
    )
    parser.add_argument("--output-dir", "--output_dir", default="analyses")
    parser.add_argument("--cluster-id", default=None)
    parser.add_argument("--process-id", default=None)
    parser.add_argument("--git-commit", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    subject_name = Path(args.subject).stem
    if args.output_prefix is None:
        suffix = f"-{args.value[0]}" if args.value and len(args.value) == 1 else ""
        output_prefix = (
            Path(args.output_dir)
            / args.analysis
            / f"{args.model}.{subject_name}.{args.analysis}{suffix}"
        )
    else:
        output_prefix = Path(args.output_prefix)

    options = training_options_for(args.model)
    started_at = datetime.now(timezone.utc)
    metadata = {
        "subject": subject_name,
        "subject_filepath": args.subject,
        "model": args.model,
        "analysis": args.analysis,
        "value_definition": (
            "waveforms averaged per example"
            if args.analysis == "subaverage"
            else "subaveraged training examples per cross-validation fold"
        ),
        "training_options": options,
        "git_commit": args.git_commit or _git_commit(),
        "condor_cluster_id": args.cluster_id,
        "condor_process_id": args.process_id,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "duration_seconds": 0.0,
    }
    # Ensure Condor always has declared output files, even if the process is
    # terminated during model training.
    write_analysis_results(
        output_prefix,
        condition_results=[],
        metadata=metadata,
    )

    results = run_analysis_conditions(
        model_name=args.model,
        subject_filepath=args.subject,
        analysis=args.analysis,
        values=args.value,
        training_options=options,
    )
    finished_at = datetime.now(timezone.utc)
    metadata["finished_at"] = finished_at.isoformat()
    metadata["duration_seconds"] = (finished_at - started_at).total_seconds()
    summary_path, predictions_path = write_analysis_results(
        output_prefix,
        condition_results=results,
        metadata=metadata,
    )
    print(f"[run_analysis] Summary written to {summary_path.resolve()}")
    print(f"[run_analysis] Predictions written to {predictions_path.resolve()}")

    failed_values = [result.value for result in results if result.status != "success"]
    if failed_values:
        raise SystemExit(f"Analysis failed for values: {failed_values}")


if __name__ == "__main__":
    main()
