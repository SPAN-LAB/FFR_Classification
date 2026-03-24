"""
SPAN Lab - FFR Classification

Filename: accuracy_against_data_amount.py
Author(s): Kevin Chen
Description: A function that evaluates a model's performance on various data amounts.
    Data amount refers to the number of trials whose data is used for training.
"""


from pathlib import Path

from .utils import get_subject_loaded_pipelines, stratified_deterministic_sample
from ..core import AnalysisPipeline
from ..constants.defaults import SUBAVERAGE_SIZE
from .iteration import iteration


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

            max_data_amount = len(pipeline.subjects[0].trials)
            for data_amount in range(min_trials, max_data_amount + 1, stride):
                reduced_trials = stratified_deterministic_sample(
                    pipeline.subjects[0],
                    data_amount
                )
                reduced_pipeline = pipeline.deepcopy()
                reduced_pipeline.subjects[0].trials = reduced_trials

                iteration(
                    model_name,
                    training_options,
                    SUBAVERAGE_SIZE,
                    reduced_pipeline,
                    write_directory,
                    pkl_filename_prefix,
                    data_amount
                )