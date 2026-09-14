"""Compatibility API for training-data-amount sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .job_result import write_analysis_results
from .runner import run_analysis_conditions


def accuracy_against_data_amount(
    min_trials: int,
    stride: int,
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, Any],
    output_folder_path: str,
    defer_subject_loading: bool = True,
):
    """Run fixed-test-set learning curves through the canonical implementation."""
    del defer_subject_loading
    for model_name in model_names:
        for subject_filepath in subject_filepaths:
            subject_name = Path(subject_filepath).stem
            results = run_analysis_conditions(
                model_name=model_name,
                subject_filepath=subject_filepath,
                analysis="data_amount",
                training_options=training_options,
                data_amount_min=min_trials,
                data_amount_stride=stride,
            )
            write_analysis_results(
                Path(output_folder_path)
                / f"{model_name}.{subject_name}.data_amount",
                condition_results=results,
                metadata={
                    "subject": subject_name,
                    "subject_filepath": subject_filepath,
                    "model": model_name,
                    "analysis": "data_amount",
                    "value_definition": (
                        "subaveraged training examples per cross-validation fold"
                    ),
                    "training_options": training_options,
                },
            )
