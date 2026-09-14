"""Canonical execution logic for one analysis sweep."""

from __future__ import annotations

import time
import traceback
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from ..core import AnalysisPipeline
from ..models.utils import find_model
from .settings import (
    ANALYSIS_TYPES,
    DATA_AMOUNT_MIN,
    DATA_AMOUNT_STRIDE,
    DATA_AMOUNT_SUBAVERAGE_SIZE,
    NUM_FOLDS,
    SUBAVERAGE_VALUES,
    TRIM_END_MS,
    TRIM_START_MS,
    training_options_for,
)


@dataclass
class ConditionResult:
    value: int | None
    status: str
    pipeline: AnalysisPipeline | None
    started_at: str
    finished_at: str
    duration_seconds: float
    error_traceback: str | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure(value: int | None, started_at: str, started_timer: float) -> ConditionResult:
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


def _strip_training_state(pipeline: AnalysisPipeline) -> None:
    """Retain labels and predictions without holding every trained model and waveform."""
    pipeline.models = []
    for subject in pipeline.subjects:
        for trial in subject.trials:
            trial.data = []
            trial.timestamps = []
            trial.features = {}


def run_analysis_conditions(
    *,
    model_name: str,
    subject_filepath: str,
    analysis: str,
    values: Iterable[int] | None = None,
    training_options: dict[str, Any] | None = None,
    data_amount_min: int = DATA_AMOUNT_MIN,
    data_amount_stride: int = DATA_AMOUNT_STRIDE,
) -> list[ConditionResult]:
    """Run selected values, or the complete default sweep when values is omitted."""
    if analysis not in ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis '{analysis}'. Expected one of {ANALYSIS_TYPES}.")
    if data_amount_min < 1:
        raise ValueError("data_amount_min must be at least 1")
    if data_amount_stride < 1:
        raise ValueError("data_amount_stride must be at least 1")
    selected_values = _normalize_values(values)
    options = dict(training_options or training_options_for(model_name))

    preparation_started_at = _timestamp()
    preparation_timer = time.monotonic()
    try:
        base_pipeline = (
            AnalysisPipeline()
            .load_subjects(subject_filepath)
            .trim_by_timestamp(start_time=TRIM_START_MS, end_time=TRIM_END_MS)
        )
        concrete_model = find_model(model_name)
        if analysis == "data_amount":
            train_parameters = inspect.signature(concrete_model.train).parameters
            accepts_trials = "trials" in train_parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in train_parameters.values()
            )
            if not accepts_trials:
                raise TypeError(
                    f"Model {model_name} cannot run data_amount because its train() "
                    "method does not accept a trial subset."
                )
            if concrete_model.needs_all_subjects:
                raise TypeError(
                    f"Model {model_name} requires multiple subjects and cannot run "
                    "the per-subject data_amount analysis."
                )
    except Exception:
        fallback_value = selected_values[0] if selected_values else None
        return [_failure(fallback_value, preparation_started_at, preparation_timer)]

    fixed_fold_pipeline = None
    if analysis == "subaverage":
        planned_values = selected_values or list(SUBAVERAGE_VALUES)
    else:
        try:
            fixed_fold_pipeline = (
                base_pipeline.deepcopy()
                .subaverage(size=DATA_AMOUNT_SUBAVERAGE_SIZE)
                .fold(num_folds=NUM_FOLDS)
            )
            folds = fixed_fold_pipeline.subjects[0].folds
            total_trials = sum(len(fold) for fold in folds)
            max_training_amount = min(total_trials - len(fold) for fold in folds)
            if selected_values is None:
                if data_amount_min > max_training_amount:
                    raise ValueError(
                        f"Minimum training amount {data_amount_min} exceeds the available "
                        f"training pool ({max_training_amount})."
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
            if analysis == "subaverage":
                pipeline = (
                    base_pipeline.deepcopy()
                    .subaverage(size=value)
                    .fold(num_folds=NUM_FOLDS)
                    .evaluate_model(
                        model_name=model_name,
                        training_options=options,
                    )
                )
            else:
                pipeline = fixed_fold_pipeline.deepcopy().evaluate_model_with_training_amount(
                    model_name=model_name,
                    training_options=options,
                    training_amount=value,
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
