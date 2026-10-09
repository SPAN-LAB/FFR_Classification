"""Canonical condition sweeps for leave-one-subject-out generic models."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..core import AnalysisPipeline, EEGSubject, EEGTrial
from ..core.utils.sampling import sds2
from .runner import ConditionResult
from .settings import (
    ANALYSIS_TYPES,
    DATA_AMOUNT_MIN,
    DATA_AMOUNT_STRIDE,
    DATA_AMOUNT_SUBAVERAGE_SIZE,
    GENERIC_NUM_EPOCHS,
    SUBAVERAGE_VALUES,
    TRIM_END_MS,
    TRIM_START_MS,
    generic_training_options_for,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure(
    value: int | None,
    started_at: str,
    started_timer: float,
) -> ConditionResult:
    return ConditionResult(
        value=value,
        status="failure",
        pipeline=None,
        started_at=started_at,
        finished_at=_timestamp(),
        duration_seconds=time.monotonic() - started_timer,
        error_traceback=traceback.format_exc(),
    )


def _normalize_values(values: Iterable[int] | None) -> list[int] | None:
    if values is None:
        return None

    normalized = []
    for value in values:
        value = int(value)
        if value < 1:
            raise ValueError("Analysis values must be at least 1")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one analysis value is required")
    return normalized


def resolve_subject_filepaths(
    paths: str | Path | Iterable[str | Path],
) -> list[str]:
    """Resolve a directory or explicit paths into a stable subject-file list."""
    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        if candidate.is_dir():
            resolved = sorted(candidate.glob("*.mat"))
        elif candidate.is_file():
            resolved = [candidate]
        else:
            raise ValueError(f"Subject path does not exist: {candidate}")
    else:
        resolved = [Path(path) for path in paths]

    if len(resolved) < 2:
        raise ValueError("Generic analysis requires at least two subject files.")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise ValueError(f"Subject files do not exist: {', '.join(missing)}")
    invalid = [str(path) for path in resolved if path.suffix.lower() != ".mat"]
    if invalid:
        raise ValueError(f"Subject files must end in .mat: {', '.join(invalid)}")

    names = [path.stem for path in resolved]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Subject filenames must have unique stems: "
            + ", ".join(duplicate_names)
        )
    return [str(path) for path in resolved]


def _subject_labels(subject) -> set[Any]:
    return {trial.label for trial in subject.trials}


def _require_complete_subject_labels(
    pipeline: AnalysisPipeline,
    expected_labels: dict[str, set[Any]],
    *,
    value: int,
) -> None:
    for subject in pipeline.subjects:
        labels = _subject_labels(subject)
        if labels != expected_labels[subject.name]:
            raise ValueError(
                f"value={value} removes every usable trial for at least one "
                f"category in subject {subject.name}."
            )


def _strip_training_state(pipeline: AnalysisPipeline) -> None:
    pipeline.models = []
    for subject in pipeline.subjects:
        subject.folds = None
        for trial in subject.trials:
            trial.data = []
            trial.timestamps = []
            trial.features = {}


def _copy_pipeline_for_condition(
    source_pipeline: AnalysisPipeline,
) -> AnalysisPipeline:
    """Copy trial metadata while sharing read-only waveform arrays."""
    pipeline = AnalysisPipeline()
    for source_subject in source_pipeline.subjects:
        subject = EEGSubject(source_filepath=source_subject.source_filepath)
        subject.labels_map = dict(source_subject.labels_map)
        for source_trial in source_subject.trials:
            trial = EEGTrial(
                subject=subject,
                data=source_trial.data,
                timestamps=source_trial.timestamps,
                trial_index=source_trial.trial_index,
                raw_label=source_trial.raw_label,
                mapped_label=source_trial.mapped_label,
            )
            subject.trials.append(trial)
        pipeline.subjects.append(subject)
    return pipeline


def run_generic_analysis_conditions(
    *,
    model_name: str,
    subject_filepaths: str | Path | Iterable[str | Path],
    held_out_subject: str,
    analysis: str,
    values: Iterable[int] | None = None,
    training_options: dict[str, Any] | None = None,
    data_var: str = "ffr_nodss",
    data_amount_min: int = DATA_AMOUNT_MIN,
    data_amount_stride: int = DATA_AMOUNT_STRIDE,
    data_amount_subaverage_size: int = DATA_AMOUNT_SUBAVERAGE_SIZE,
) -> list[ConditionResult]:
    """Run a generic LOSO sweep for one fixed held-out subject."""
    if analysis not in ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis {analysis!r}. Expected one of {ANALYSIS_TYPES}.")
    if data_amount_min < 1:
        raise ValueError("data_amount_min must be at least 1")
    if data_amount_stride < 1:
        raise ValueError("data_amount_stride must be at least 1")
    if data_amount_subaverage_size < 1:
        raise ValueError("data_amount_subaverage_size must be at least 1")

    selected_values = _normalize_values(values)
    options = generic_training_options_for(model_name)
    if training_options is not None:
        options.update(training_options)
    options["num_epochs"] = GENERIC_NUM_EPOCHS
    options["validation_ratio"] = 0.0
    held_out_name = Path(held_out_subject).stem

    preparation_started_at = _timestamp()
    preparation_timer = time.monotonic()
    try:
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
        expected_labels = {
            subject.name: _subject_labels(subject)
            for subject in base_pipeline.subjects
        }
        if any(not labels for labels in expected_labels.values()):
            raise ValueError("Every generic-analysis subject must contain labeled trials.")

        if analysis == "subaverage":
            planned_values = selected_values or list(SUBAVERAGE_VALUES)
        else:
            max_training_amount = min(
                len(subject.trials)
                for subject in base_pipeline.subjects
            )
            if selected_values is None:
                if data_amount_min > max_training_amount:
                    raise ValueError(
                        f"Minimum training amount {data_amount_min} exceeds the "
                        f"smallest training-subject pool ({max_training_amount})."
                    )
                planned_values = list(
                    range(
                        data_amount_min,
                        max_training_amount + 1,
                        data_amount_stride,
                    )
                )
            else:
                planned_values = selected_values
    except Exception:
        fallback_value = selected_values[0] if selected_values else None
        return [_failure(fallback_value, preparation_started_at, preparation_timer)]

    results = []
    for value in planned_values:
        started_at = _timestamp()
        started_timer = time.monotonic()
        try:
            pipeline = _copy_pipeline_for_condition(base_pipeline)
            if analysis == "subaverage":
                pipeline.subaverage(size=value)
            else:
                for subject in pipeline.subjects:
                    if subject.name != held_out_name:
                        if value > len(subject.trials):
                            raise ValueError(
                                f"training amount {value} exceeds the available "
                                f"trials ({len(subject.trials)}) for {subject.name}."
                            )
                        subject.trials = list(sds2(list(subject.trials), value))
                    subject.subaverage(data_amount_subaverage_size)

            _require_complete_subject_labels(
                pipeline,
                expected_labels,
                value=value,
            )
            pipeline.evaluate_generic_model(
                model_name=model_name,
                training_options=options,
                only_held_out=[held_out_name],
            )
            _strip_training_state(pipeline)
            results.append(
                ConditionResult(
                    value=value,
                    status="success",
                    pipeline=pipeline,
                    started_at=started_at,
                    finished_at=_timestamp(),
                    duration_seconds=time.monotonic() - started_timer,
                )
            )
        except Exception:
            results.append(_failure(value, started_at, started_timer))

    return results
