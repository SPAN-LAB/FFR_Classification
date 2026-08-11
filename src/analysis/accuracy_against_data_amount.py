"""
SPAN Lab - FFR Classification

Filename: accuracy_against_data_amount.py
Author(s): Kevin Chen
Description: A function that evaluates a model's performance on various data amounts.
    Data amount refers to the number of trials whose data is used for training.
"""


from pathlib import Path

from .utils import get_subject_loaded_pipelines
from ..core import AnalysisPipeline
from ..constants.defaults import (
    NUM_FOLDS,
    SUBAVERAGE_SIZE,
    TRIM_END_TIME,
    TRIM_START_TIME,
)
from .iteration import training_amount_iteration


def accuracy_against_data_amount(
    min_trials: int,
    stride: int,
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    defer_subject_loading: bool = True
):

    # Setting up variables and time keepers

    independent_var_name = "data_amount"
    pkl_filename_prefix = f"{independent_var_name}"

    # If don't defer subject loading, load all the subjects now
    subject_loaded_pipelines = None
    if not defer_subject_loading:
        subject_loaded_pipelines = get_subject_loaded_pipelines(subject_filepaths)

    for model_name in model_names:
        for subject_filepath in subject_filepaths:

            # The base subject pipeline state used for this subject.
            # Do not modify, only deeply copy.
            if not defer_subject_loading:
                pipeline = subject_loaded_pipelines[subject_filepath]
            else:
                pipeline = AnalysisPipeline().load_subjects(subject_filepath)

            subject_filename = Path(subject_filepath).stem
            write_directory = (
                Path(output_folder_path)
                / independent_var_name
                / model_name
                / subject_filename
            )

            # Build the folds once. Every data-amount condition below receives a
            # deep copy of these exact folds, so its test trials never change.
            fixed_fold_pipeline = (
                pipeline.deepcopy()
                .trim_by_timestamp(
                    start_time=TRIM_START_TIME,
                    end_time=TRIM_END_TIME,
                )
                .subaverage(size=SUBAVERAGE_SIZE)
                .fold(num_folds=NUM_FOLDS)
            )
            folds = fixed_fold_pipeline.subjects[0].folds
            total_trials = sum(len(fold) for fold in folds)
            max_data_amount = min(total_trials - len(fold) for fold in folds)
            if min_trials > max_data_amount:
                raise ValueError(
                    f"min_trials={min_trials} exceeds the smallest training pool "
                    f"({max_data_amount}) after subaveraging and folding "
                    f"subject {subject_filename}."
                )

            for data_amount in range(min_trials, max_data_amount + 1, stride):
                training_amount_iteration(
                    model_name=model_name,
                    training_options=training_options,
                    training_amount=data_amount,
                    pipeline_copy=fixed_fold_pipeline.deepcopy(),
                    write_directory=write_directory,
                    pkl_filename_prefix=pkl_filename_prefix,
                )
