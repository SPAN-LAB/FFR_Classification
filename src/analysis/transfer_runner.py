"""Transfer analysis: pre-train on every other subject, then fine-tune on the held-out subject."""

from __future__ import annotations

import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core import AnalysisPipeline, EEGSubject, EEGTrial
from ..core.utils.sampling import sample_balanced_trials
from ..models.utils import find_model
from .generic_runner import resolve_subject_filepaths
from .runner import ConditionResult
from .settings import (
    DATA_AMOUNT_SAMPLING_SEED,
    NUM_FOLDS,
    TRANSFER_CALIBRATION_VALUES,
    TRANSFER_FINE_TUNE_EPOCHS,
    TRANSFER_FINE_TUNE_LR_SCALE,
    TRANSFER_MODEL_NAMES,
    TRANSFER_SUBAVERAGE_SIZE,
    TRIM_END_MS,
    TRIM_START_MS,
    generic_training_options_for,
)
from .transfer_checkpoint import (
    CHECKPOINT_MATCH_FIELDS,
    PretrainedCheckpoint,
    jsonable,
)


# Value 0 is the pre-trained model without fine-tuning: this protocol's baseline.
BASELINE_VALUE = 0


@dataclass
class TransferRunResult:
    checkpoint: PretrainedCheckpoint
    conditions: list[ConditionResult]
    fine_tune_options: dict[str, Any] | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure(
    value: int | None,
    started_at: str,
    started_timer: float,
    condition_metadata: dict[str, Any] | None = None,
    error_traceback: str | None = None,
) -> ConditionResult:
    return ConditionResult(
        value=value,
        status="failure",
        pipeline=None,
        started_at=started_at,
        finished_at=_timestamp(),
        duration_seconds=time.monotonic() - started_timer,
        error_traceback=error_traceback or traceback.format_exc(),
        condition_metadata=condition_metadata,
    )


def _normalize_values(values: Iterable[int] | None) -> list[int]:
    if values is None:
        return list(TRANSFER_CALIBRATION_VALUES)

    normalized = []
    for value in values:
        value = int(value)
        if value < 1:
            raise ValueError(
                "Calibration values must be at least 1; value 0 (the pre-trained "
                "baseline) always runs."
            )
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one calibration value is required")
    return normalized


def _label_counts(subject: EEGSubject) -> dict[Any, int]:
    return {
        label: len(trials)
        for label, trials in subject.grouped_trials().items()
    }


def _training_labels_map(training_subjects: list[EEGSubject]) -> dict[Any, int]:
    """Index labels in order of first appearance, as generic analysis does."""
    label_order = []
    seen_labels = set()
    for subject in training_subjects:
        for trial in subject.trials:
            if trial.label not in seen_labels:
                seen_labels.add(trial.label)
                label_order.append(trial.label)
    return {label: index for index, label in enumerate(label_order)}


def _check_model_name(model_name: str) -> None:
    if model_name.lower() not in {name.lower() for name in TRANSFER_MODEL_NAMES}:
        raise ValueError(
            f"Model {model_name!r} is not configured for transfer analysis. "
            f"Expected one of: {', '.join(TRANSFER_MODEL_NAMES)}."
        )


def _check_supported(model_name: str, concrete_model: type, options: dict[str, Any]) -> None:
    from ..models.utils.torchnn_base import TorchNNBase

    if not issubclass(concrete_model, TorchNNBase):
        raise ValueError(
            f"{model_name} cannot be fine-tuned; transfer analysis supports "
            "PyTorch models only."
        )
    required_inputs = concrete_model.required_inputs_for_options(options)
    if required_inputs != ["raw"]:
        raise ValueError(
            "Transfer analysis supports raw-waveform models only; "
            f"{model_name} requires {required_inputs}."
        )
    if concrete_model.needs_all_subjects:
        raise ValueError(
            f"Transfer analysis does not support the specialized training "
            f"contract used by {model_name}."
        )


def _subaveraged_subject(
    trials: list[EEGTrial],
    labels_map: dict[Any, int],
    size: int,
    source_filepath: str,
) -> EEGSubject:
    """Subaverage trials into a new subject that uses the pre-training label map."""
    subject = EEGSubject(trials=list(trials), source_filepath=source_filepath)
    subject.subaverage(size)
    subject.labels_map = dict(labels_map)
    missing = set(labels_map) - {trial.label for trial in subject.trials}
    if missing:
        raise ValueError(
            f"Subaveraging {len(trials)} trials at size {size} leaves no complete "
            f"example for labels {sorted(map(str, missing))}."
        )
    return subject


def _model_from_checkpoint(
    concrete_model: type,
    state_dict: dict[str, Any],
    options: dict[str, Any],
    subject: EEGSubject,
):
    model = concrete_model(options)
    model.set_subject(subject)
    model.build()
    model.model.load_state_dict(state_dict)
    model.model.to(model.device)
    return model


def _pretrain(
    concrete_model: type,
    pool: EEGSubject,
    options: dict[str, Any],
) -> dict[str, Any]:
    model = concrete_model(options)
    model.set_subject(pool)
    model.train(trials=list(pool.trials), validation_trials=None)
    return {
        key: value.detach().cpu().clone()
        for key, value in model.model.state_dict().items()
    }


def _mismatched_fields(
    checkpoint: PretrainedCheckpoint,
    expected: dict[str, Any],
) -> list[str]:
    return [
        field
        for field in CHECKPOINT_MATCH_FIELDS
        if jsonable(checkpoint.metadata.get(field)) != jsonable(expected[field])
    ]


def _condition_pipeline(
    held_out: EEGSubject,
    labels_map: dict[Any, int],
    evaluated_folds: list[list[EEGTrial]],
) -> AnalysisPipeline:
    """Hold the evaluated test folds so the result writer can record predictions."""
    subject = EEGSubject(source_filepath=held_out.source_filepath)
    subject.labels_map = dict(labels_map)
    subject.folds = evaluated_folds
    subject.trials = [trial for fold in evaluated_folds for trial in fold]
    for trial in subject.trials:
        trial.data = []
        trial.timestamps = []
        trial.features = {}
    pipeline = AnalysisPipeline()
    pipeline.subjects = [subject]
    return pipeline


def run_transfer_analysis_conditions(
    *,
    model_name: str,
    subject_filepaths: str | Path | Iterable[str | Path],
    held_out_subject: str,
    values: Iterable[int] | None = None,
    data_var: str = "ffr_nodss",
    pretrained: PretrainedCheckpoint | None = None,
    on_checkpoint: Callable[[PretrainedCheckpoint], None] | None = None,
) -> TransferRunResult:
    """Pre-train on every other subject (or validate `pretrained`), report the
    checkpoint through `on_checkpoint`, then fine-tune on N calibration trials
    per fold of the held-out subject. Value 0 is the pre-trained baseline."""
    source = "trained" if pretrained is None else "loaded"
    held_out_name = Path(held_out_subject).stem
    size = TRANSFER_SUBAVERAGE_SIZE
    checkpoint_reported = False

    def report(checkpoint: PretrainedCheckpoint) -> PretrainedCheckpoint:
        # The CLI writes the checkpoint files here, exactly once, before fine-tuning.
        nonlocal checkpoint_reported
        if not checkpoint_reported:
            checkpoint_reported = True
            if on_checkpoint is not None:
                on_checkpoint(checkpoint)
        return checkpoint

    def failed(
        metadata: dict[str, Any],
        error_traceback: str,
        started_at: str,
        started_timer: float,
    ) -> TransferRunResult:
        checkpoint = report(
            PretrainedCheckpoint(
                status="failure",
                source=source,
                state_dict=None,
                metadata=metadata,
                source_path=pretrained.source_path if pretrained else None,
                source_git_commit=pretrained.source_git_commit if pretrained else None,
                error_traceback=error_traceback,
            )
        )
        return TransferRunResult(
            checkpoint=checkpoint,
            conditions=[
                _failure(None, started_at, started_timer, error_traceback=error_traceback)
            ],
        )

    preparation_started_at = _timestamp()
    preparation_timer = time.monotonic()
    expected: dict[str, Any] = {"model": model_name, "held_out_subject": held_out_name}
    try:
        planned_values = _normalize_values(values)
        _check_model_name(model_name)
        requested_options = generic_training_options_for(model_name)
        concrete_model = find_model(model_name)
        _check_supported(model_name, concrete_model, requested_options)

        paths = resolve_subject_filepaths(subject_filepaths)
        base_pipeline = (
            AnalysisPipeline()
            .load_subjects(paths, data_var=data_var)
            .trim_by_timestamp(start_time=TRIM_START_MS, end_time=TRIM_END_MS)
        )
        subjects_by_name = {subject.name: subject for subject in base_pipeline.subjects}
        if held_out_name not in subjects_by_name:
            raise ValueError(
                f"Held-out subject {held_out_name!r} was not found. "
                f"Available subjects: {', '.join(sorted(subjects_by_name))}."
            )
        held_out = subjects_by_name[held_out_name]
        training_subjects = [
            subject
            for subject in base_pipeline.subjects
            if subject.name != held_out_name
        ]
        if not training_subjects:
            raise ValueError("Transfer analysis needs at least one pre-training subject.")
        if any(not subject.trials for subject in [held_out, *training_subjects]):
            raise ValueError("Every transfer-analysis subject must contain trials.")

        labels_map = _training_labels_map(training_subjects)
        held_out_labels = {trial.label for trial in held_out.trials}
        if held_out_labels != set(labels_map):
            raise ValueError(
                f"Held-out subject {held_out_name} has a different label set. "
                f"Missing labels: {sorted(map(str, set(labels_map) - held_out_labels))}; "
                f"unseen labels: {sorted(map(str, held_out_labels - set(labels_map)))}."
            )
        waveform_sizes = {
            len(trial.data)
            for subject in base_pipeline.subjects
            for trial in subject.trials
        }
        if len(waveform_sizes) != 1:
            raise ValueError(
                "Transfer analysis requires every waveform to have the same length; "
                f"found {sorted(waveform_sizes)}."
            )
        n_times = waveform_sizes.pop()

        label_counts = {
            subject.name: _label_counts(subject)
            for subject in training_subjects
        }
        pretraining_examples = sum(
            count // size
            for counts in label_counts.values()
            for count in counts.values()
        )
        pretraining_options = dict(requested_options)
        pretraining_options["n_classes"] = len(labels_map)
        pretraining_options["n_times"] = n_times
        pretraining_options["validation_ratio"] = 0.0

        expected = {
            "model": model_name,
            "held_out_subject": held_out_name,
            "pretraining_subjects": sorted(label_counts),
            "pretraining_label_counts": label_counts,
            "labels_map": labels_map,
            "n_classes": len(labels_map),
            "n_times": n_times,
            "pretraining_options": pretraining_options,
            "subaverage_size": size,
            "trim_start_ms": TRIM_START_MS,
            "trim_end_ms": TRIM_END_MS,
            "data_var": data_var,
            "pretraining_examples": pretraining_examples,
        }
        if held_out_name in expected["pretraining_subjects"]:
            raise ValueError(f"Held-out subject {held_out_name} is in the pre-training set.")

        # Fold the raw held-out trials once; every value uses these test folds.
        folds = held_out.fold(num_folds=NUM_FOLDS).folds
        counts_by_fold = [Counter(trial.label for trial in fold) for fold in folds]
        total_counts = sum(counts_by_fold, Counter())
        max_per_label = min(
            total_counts[label] - fold_counts[label]
            for fold_counts in counts_by_fold
            for label in labels_map
        )
        max_calibration_trials = len(labels_map) * max_per_label
        too_large = [value for value in planned_values if value > max_calibration_trials]
        if too_large:
            raise ValueError(
                f"Calibration values {too_large} exceed the balanced calibration pool "
                f"of {held_out_name} ({max_calibration_trials} raw trials per fold)."
            )
    except Exception:
        return failed(expected, traceback.format_exc(), preparation_started_at, preparation_timer)

    if pretrained is not None:
        mismatches = _mismatched_fields(pretrained, expected)
        if pretrained.status != "success" or pretrained.state_dict is None:
            mismatches.insert(0, "status")
        if mismatches:
            message = (
                "Loaded checkpoint does not match this run on: "
                + ", ".join(mismatches)
                + ". Pre-training was not run as a fallback."
            )
            return failed(expected, message, preparation_started_at, preparation_timer)
        checkpoint = PretrainedCheckpoint(
            status="success",
            source="loaded",
            state_dict=pretrained.state_dict,
            metadata=expected,
            source_path=pretrained.source_path,
            source_git_commit=pretrained.source_git_commit,
        )
    else:
        try:
            for subject in training_subjects:
                subject.subaverage(size)
            pool = EEGSubject.create_merged(subjects=training_subjects)
            pool.source_filepath = "<transfer-pretraining-pool>.mat"
            pool.labels_map = dict(labels_map)
            if len(pool.trials) != pretraining_examples:
                raise ValueError(
                    f"Expected {pretraining_examples} pre-training examples; "
                    f"found {len(pool.trials)}."
                )
            state_dict = _pretrain(concrete_model, pool, pretraining_options)
            del pool
            checkpoint = PretrainedCheckpoint(
                status="success",
                source="trained",
                state_dict=state_dict,
                metadata=expected,
            )
        except Exception:
            return failed(expected, traceback.format_exc(), preparation_started_at, preparation_timer)

    # Release the pre-training subjects; only the held-out subject is needed now.
    for subject in training_subjects:
        subject.trials = []
    base_pipeline.subjects = [held_out]
    report(checkpoint)

    fine_tune_options = dict(pretraining_options)
    fine_tune_options["learning_rate"] = (
        float(pretraining_options["learning_rate"]) * TRANSFER_FINE_TUNE_LR_SCALE
    )
    fine_tune_options["num_epochs"] = TRANSFER_FINE_TUNE_EPOCHS
    fine_tune_options["validation_ratio"] = 0.0

    results = []
    for value in [BASELINE_VALUE, *planned_values]:
        started_at = _timestamp()
        started_timer = time.monotonic()
        is_baseline = value == BASELINE_VALUE
        condition_metadata: dict[str, Any] = {
            "condition": "pretrained_baseline" if is_baseline else "fine_tuned",
            "fine_tuned": not is_baseline,
            "raw_calibration_trials": value,
        }
        try:
            evaluated_folds = []
            calibration_examples = []
            test_examples = []
            for fold_i, raw_test_trials in enumerate(folds):
                test_subject = _subaveraged_subject(
                    raw_test_trials, labels_map, size, held_out.source_filepath
                )
                if is_baseline:
                    model = _model_from_checkpoint(
                        concrete_model,
                        checkpoint.state_dict,
                        pretraining_options,
                        test_subject,
                    )
                else:
                    calibration_pool = [
                        trial
                        for pool_i, fold_trials in enumerate(folds)
                        if pool_i != fold_i
                        for trial in fold_trials
                    ]
                    # Same seed as per-subject data amount, so equal N picks equal trials.
                    sampled = sample_balanced_trials(
                        calibration_pool,
                        value,
                        seed=f"data-amount:{DATA_AMOUNT_SAMPLING_SEED}:{held_out.name}:{fold_i}",
                    )
                    test_indices = {trial.trial_index for trial in raw_test_trials}
                    if any(trial.trial_index in test_indices for trial in sampled):
                        raise ValueError(
                            f"Calibration trials overlap test fold {fold_i} of {held_out.name}."
                        )
                    calibration_subject = _subaveraged_subject(
                        sampled, labels_map, size, held_out.source_filepath
                    )
                    model = _model_from_checkpoint(
                        concrete_model,
                        checkpoint.state_dict,
                        fine_tune_options,
                        calibration_subject,
                    )
                    model.train(
                        trials=list(calibration_subject.trials),
                        validation_trials=None,
                        rebuild=False,
                    )
                    calibration_examples.append(len(calibration_subject.trials))

                model.infer(trials=test_subject.trials)
                evaluated_folds.append(test_subject.trials)
                test_examples.append(len(test_subject.trials))
                del model

            condition_metadata["test_examples_per_fold"] = test_examples
            if not is_baseline:
                condition_metadata["calibration_examples_per_fold"] = calibration_examples
            accuracy = EEGTrial.get_accuracy(evaluated_folds)
            print(
                f"Transfer accuracy on {held_out.name} with {value} raw calibration "
                f"trials per fold: {accuracy}"
            )
            results.append(
                ConditionResult(
                    value=value,
                    status="success",
                    pipeline=_condition_pipeline(held_out, labels_map, evaluated_folds),
                    started_at=started_at,
                    finished_at=_timestamp(),
                    duration_seconds=time.monotonic() - started_timer,
                    condition_metadata=condition_metadata,
                )
            )
        except Exception:
            results.append(_failure(value, started_at, started_timer, condition_metadata))

    return TransferRunResult(
        checkpoint=checkpoint,
        conditions=results,
        fine_tune_options=fine_tune_options,
    )
