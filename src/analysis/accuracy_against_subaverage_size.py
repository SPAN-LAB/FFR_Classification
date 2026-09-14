"""Compatibility API for subaverage-size sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .job_result import write_analysis_results
from .runner import run_analysis_conditions


def accuracy_against_subaverage_size(
    subaverage_sizes: list[int],
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, Any],
    output_folder_path: str,
    include_null_case: bool = True,
    defer_subject_loading: bool = True,
):
    """Run subaverage sweeps through the canonical analysis implementation."""
    del defer_subject_loading
    values = list(dict.fromkeys(subaverage_sizes))
    if include_null_case and 1 not in values:
        values.insert(0, 1)

    for model_name in model_names:
        for subject_filepath in subject_filepaths:
            subject_name = Path(subject_filepath).stem
            results = run_analysis_conditions(
                model_name=model_name,
                subject_filepath=subject_filepath,
                analysis="subaverage",
                values=values,
                training_options=training_options,
            )
            write_analysis_results(
                Path(output_folder_path)
                / f"{model_name}.{subject_name}.subaverage",
                condition_results=results,
                metadata={
                    "subject": subject_name,
                    "subject_filepath": subject_filepath,
                    "model": model_name,
                    "analysis": "subaverage",
                    "value_definition": "waveforms averaged per example",
                    "training_options": training_options,
                },
            )
