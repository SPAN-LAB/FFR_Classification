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
    DATA_AMOUNT_SAMPLING_SEED,
    DATA_AMOUNT_SUBAVERAGE_SIZE,
    GENERIC_ANALYSIS_TYPES,
    GENERIC_DATA_AMOUNT_MAX_EXCLUDED_TRAINING_SUBJECTS,
    GENERIC_DATA_AMOUNT_MIN,
    GENERIC_DATA_AMOUNT_SAMPLING_SEED,
    GENERIC_DATA_AMOUNT_STRIDE,
    GENERIC_MODEL_NAMES,
    NUM_FOLDS,
    TRANSFER_ANALYSIS,
    TRANSFER_CALIBRATION_VALUES,
    TRANSFER_FINE_TUNE_EPOCHS,
    TRANSFER_FINE_TUNE_LR_SCALE,
    TRANSFER_SUBAVERAGE_SIZE,
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
    parser.add_argument("--analysis", required=True, choices=GENERIC_ANALYSIS_TYPES)
    parser.add_argument(
        "--value",
        action="append",
        type=int,
        help=(
            "Run only this value. May be supplied more than once. For transfer, "
            "the value is raw calibration trials per fold; the baseline (0) "
            "always runs."
        ),
    )
    parser.add_argument(
        "--pretrained-weights",
        default=None,
        help=(
            "Transfer only: load this pre-trained checkpoint (.pt, .json, or "
            "their prefix) instead of pre-training"
        ),
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


def _main_transfer(args: argparse.Namespace) -> None:
    """Run the transfer analysis; the CLI owns all checkpoint file I/O."""
    import traceback

    from src.analysis.runner import ConditionResult
    from src.analysis.transfer_checkpoint import (
        CheckpointError,
        PretrainedCheckpoint,
        read_checkpoint,
        write_checkpoint,
    )
    from src.analysis.transfer_runner import run_transfer_analysis_conditions

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

    if args.output_prefix is None:
        # Runs of selected values carry them in the name, so they never
        # overwrite full runs or each other.
        output_dir = "transfer_partial" if args.value else "transfer"
        suffix = f".v{'-'.join(map(str, args.value))}" if args.value else ""
        output_prefix = (
            Path(args.output_dir)
            / "generic"
            / output_dir
            / args.model
            / f"{args.model}.{held_out_filename}.generic.transfer{suffix}"
        )
    else:
        output_prefix = Path(args.output_prefix)

    commit = args.git_commit or _git_commit()
    started_at = datetime.now(timezone.utc)
    metadata = {
        "subject": held_out_name,
        "held_out_subject": held_out_name,
        "training_subjects": training_subjects,
        "subject_filepaths": subject_filepaths,
        "model": args.model,
        "analysis": TRANSFER_ANALYSIS,
        "evaluation": "generic_transfer",
        "value_definition": (
            "raw calibration trials per cross-validation fold from the held-out "
            "subject before subaveraging"
        ),
        "baseline_value": 0,
        "baseline_definition": (
            "pre-trained model without fine-tuning, evaluated on this protocol's "
            f"{NUM_FOLDS} test folds, each subaveraged separately"
        ),
        "baseline_note": (
            "not identical to generic LOSO results, which subaverage the held-out "
            "subject as a whole"
        ),
        "training_options": generic_training_options_for(args.model),
        "pretraining_options": None,
        "fine_tune_options": None,
        "fine_tune_learning_rate_scale": TRANSFER_FINE_TUNE_LR_SCALE,
        "fine_tune_num_epochs": TRANSFER_FINE_TUNE_EPOCHS,
        "subaverage_size": TRANSFER_SUBAVERAGE_SIZE,
        "num_folds": NUM_FOLDS,
        "sampling": "seeded_balanced_per_label",
        "sampling_seed": DATA_AMOUNT_SAMPLING_SEED,
        "calibration_values": list(args.value or TRANSFER_CALIBRATION_VALUES),
        "data_variable": args.data_var,
        "pretraining_examples": None,
        "checkpoint_status": None,
        "checkpoint_source": "loaded" if args.pretrained_weights else "trained",
        "checkpoint_source_path": args.pretrained_weights,
        "checkpoint_source_git_commit": None,
        "git_commit": commit,
        "condor_cluster_id": args.cluster_id,
        "condor_process_id": args.process_id,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "duration_seconds": 0.0,
    }

    def finish(condition_results: list) -> tuple[Path, Path]:
        finished_at = datetime.now(timezone.utc)
        metadata["finished_at"] = finished_at.isoformat()
        metadata["duration_seconds"] = (finished_at - started_at).total_seconds()
        return write_analysis_results(
            output_prefix,
            condition_results=condition_results,
            metadata=metadata,
        )

    def save_checkpoint(checkpoint: PretrainedCheckpoint) -> None:
        metadata["checkpoint_status"] = checkpoint.status
        metadata["checkpoint_source"] = checkpoint.source
        metadata["checkpoint_source_path"] = checkpoint.source_path
        metadata["checkpoint_source_git_commit"] = (
            checkpoint.source_git_commit if checkpoint.source == "loaded" else commit
        )
        weights_path, metadata_path = write_checkpoint(
            checkpoint,
            output_prefix,
            git_commit=commit,
        )
        print(f"[run_generic_analysis] Checkpoint ({checkpoint.status}) written to {weights_path.resolve()}")
        print(f"[run_generic_analysis] Checkpoint metadata written to {metadata_path.resolve()}")

    pretrained = None
    if args.pretrained_weights is not None:
        try:
            pretrained = read_checkpoint(args.pretrained_weights)
        except CheckpointError as error:
            error_traceback = traceback.format_exc()
            save_checkpoint(
                PretrainedCheckpoint(
                    status="failure",
                    source="loaded",
                    state_dict=None,
                    metadata={"model": args.model, "held_out_subject": held_out_name},
                    source_path=str(args.pretrained_weights),
                    error_traceback=error_traceback,
                )
            )
            now = datetime.now(timezone.utc).isoformat()
            finish([
                ConditionResult(
                    value=None,
                    status="failure",
                    pipeline=None,
                    started_at=now,
                    finished_at=now,
                    duration_seconds=0.0,
                    error_traceback=error_traceback,
                )
            ])
            raise SystemExit(f"Cannot load pre-trained checkpoint: {error}")

    write_analysis_results(output_prefix, condition_results=[], metadata=metadata)

    result = run_transfer_analysis_conditions(
        model_name=args.model,
        subject_filepaths=subject_filepaths,
        held_out_subject=held_out_name,
        values=args.value,
        data_var=args.data_var,
        pretrained=pretrained,
        on_checkpoint=save_checkpoint,
    )
    metadata["pretraining_options"] = result.checkpoint.metadata.get("pretraining_options")
    metadata["pretraining_examples"] = result.checkpoint.metadata.get("pretraining_examples")
    metadata["fine_tune_options"] = result.fine_tune_options
    summary_path, predictions_path = finish(result.conditions)
    print(f"[run_generic_analysis] Summary written to {summary_path.resolve()}")
    print(f"[run_generic_analysis] Predictions written to {predictions_path.resolve()}")

    if result.checkpoint.status != "success":
        raise SystemExit("Transfer analysis failed before fine-tuning; see the checkpoint JSON.")
    failed_values = [
        condition.value
        for condition in result.conditions
        if condition.status != "success"
    ]
    if failed_values:
        raise SystemExit(f"Transfer analysis failed for values: {failed_values}")


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
    if args.pretrained_weights is not None and args.analysis != TRANSFER_ANALYSIS:
        raise ValueError("--pretrained-weights is only valid with --analysis transfer")
    if args.analysis == TRANSFER_ANALYSIS:
        _main_transfer(args)
        return
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
            else "raw training trials per category per training subject before subaveraging"
        ),
        "training_options": options,
        "data_variable": args.data_var,
        "data_amount_subaverage_size": (
            data_amount_subaverage_size
            if args.analysis == "data_amount"
            else None
        ),
        "data_amount_sampling_seed": (
            GENERIC_DATA_AMOUNT_SAMPLING_SEED
            if args.analysis == "data_amount"
            else None
        ),
        "data_amount_min": (
            GENERIC_DATA_AMOUNT_MIN if args.analysis == "data_amount" else None
        ),
        "data_amount_stride": (
            GENERIC_DATA_AMOUNT_STRIDE if args.analysis == "data_amount" else None
        ),
        "max_excluded_training_subjects": (
            GENERIC_DATA_AMOUNT_MAX_EXCLUDED_TRAINING_SUBJECTS
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
