"""Run one generic LOSO analysis sweep for a fixed held-out subject."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.generic_runner import (
    resolve_subject_filepaths,
    run_generic_analysis_conditions,
)
from src.analysis.job_result import write_analysis_results
from src.analysis.settings import (
    ANALYSIS_TYPES,
    DATA_AMOUNT_SUBAVERAGE_SIZE,
    GENERIC_MODEL_NAMES,
    generic_training_options_for,
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
        description="Run a generic leave-one-subject-out FFR analysis."
    )
    parser.add_argument("--model", required=True, choices=GENERIC_MODEL_NAMES)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing every subject .mat file",
    )
    parser.add_argument(
        "--held-out",
        required=True,
        help="Held-out subject filename or stem, for example 4T1002.mat",
    )
    parser.add_argument("--analysis", required=True, choices=ANALYSIS_TYPES)
    parser.add_argument(
        "--value",
        action="append",
        type=int,
        help="Run only this value. May be supplied more than once.",
    )
    parser.add_argument("--data-var", default="ffr_nodss")
    parser.add_argument(
        "--data-amount-subaverage-size",
        type=int,
        default=None,
        help=(
            "Subaverage size used by data_amount analysis "
            f"(default: {DATA_AMOUNT_SUBAVERAGE_SIZE})"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Path prefix for the summary JSON and predictions CSV",
    )
    parser.add_argument("--output-dir", default="analyses")
    parser.add_argument("--cluster-id", default=None)
    parser.add_argument("--process-id", default=None)
    parser.add_argument("--git-commit", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.data_amount_subaverage_size is not None:
        if args.analysis != "data_amount":
            raise ValueError(
                "--data-amount-subaverage-size is only valid with "
                "--analysis data_amount"
            )
        if args.data_amount_subaverage_size < 1:
            raise ValueError("--data-amount-subaverage-size must be at least 1")
    data_amount_subaverage_size = (
        args.data_amount_subaverage_size
        if args.data_amount_subaverage_size is not None
        else DATA_AMOUNT_SUBAVERAGE_SIZE
    )
    subject_filepaths = resolve_subject_filepaths(args.data_dir)
    held_out_name = Path(args.held_out).stem
    matching_paths = [
        path
        for path in subject_filepaths
        if Path(path).stem == held_out_name
    ]
    if not matching_paths:
        raise ValueError(
            f"Held-out subject {held_out_name!r} was not found in {args.data_dir}."
        )

    held_out_filename = Path(matching_paths[0]).name
    training_subjects = [
        Path(path).stem
        for path in subject_filepaths
        if Path(path).stem != held_out_name
    ]
    suffix = f"-{args.value[0]}" if args.value and len(args.value) == 1 else ""
    if args.output_prefix is None:
        if args.analysis == "data_amount" and args.data_amount_subaverage_size is not None:
            output_prefix = (
                Path(args.output_dir)
                / "generic"
                / "data_amount_by_subaverage"
                / f"subaverage_{data_amount_subaverage_size}"
                / args.model
                / (
                    f"{args.model}.{held_out_filename}.generic.data_amount."
                    f"sa{data_amount_subaverage_size}{suffix}"
                )
            )
        else:
            output_prefix = (
                Path(args.output_dir)
                / "generic"
                / args.analysis
                / args.model
                / (
                    f"{args.model}.{held_out_filename}.generic."
                    f"{args.analysis}{suffix}"
                )
            )
    else:
        output_prefix = Path(args.output_prefix)

    options = generic_training_options_for(args.model)
    started_at = datetime.now(timezone.utc)
    metadata = {
        "subject": held_out_name,
        "held_out_subject": held_out_name,
        "training_subjects": training_subjects,
        "subject_filepaths": subject_filepaths,
        "model": args.model,
        "analysis": args.analysis,
        "evaluation": "generic_loso",
        "value_definition": (
            "waveforms averaged per example within each subject and category"
            if args.analysis == "subaverage"
            else "raw training trials per training subject before subaveraging"
        ),
        "training_options": options,
        "data_variable": args.data_var,
        "data_amount_subaverage_size": (
            data_amount_subaverage_size
            if args.analysis == "data_amount"
            else None
        ),
        "git_commit": args.git_commit or _git_commit(),
        "condor_cluster_id": args.cluster_id,
        "condor_process_id": args.process_id,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "duration_seconds": 0.0,
    }
    write_analysis_results(
        output_prefix,
        condition_results=[],
        metadata=metadata,
    )

    results = run_generic_analysis_conditions(
        model_name=args.model,
        subject_filepaths=subject_filepaths,
        held_out_subject=held_out_name,
        analysis=args.analysis,
        values=args.value,
        training_options=options,
        data_var=args.data_var,
        data_amount_subaverage_size=data_amount_subaverage_size,
    )
    finished_at = datetime.now(timezone.utc)
    metadata["finished_at"] = finished_at.isoformat()
    metadata["duration_seconds"] = (finished_at - started_at).total_seconds()
    summary_path, predictions_path = write_analysis_results(
        output_prefix,
        condition_results=results,
        metadata=metadata,
    )
    print(f"[run_generic_analysis] Summary written to {summary_path.resolve()}")
    print(f"[run_generic_analysis] Predictions written to {predictions_path.resolve()}")

    failed_values = [result.value for result in results if result.status != "success"]
    if failed_values:
        raise SystemExit(f"Generic analysis failed for values: {failed_values}")


if __name__ == "__main__":
    main()
