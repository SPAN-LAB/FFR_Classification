"""
SPAN Lab - FFR Classification

Filename: accuracy_against_subaverage_size.py
Author(s): Kevin Chen
Description: A function that evaluates a model's performance on various subaverage sizes.
"""


from pathlib import Path

from .utils import get_subject_loaded_pipelines
from .iteration import iteration
from ..core import AnalysisPipeline


def accuracy_against_subaverage_size(
    subaverage_sizes: list[int],
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    include_null_case: bool = True,
    defer_subject_loading: bool = True
):
    
    # Setting up variables and time keepers

    independent_var_name = "subaverage_size"
    pkl_filename_prefix = f"{independent_var_name}"
    
    # Include a case where no subaveraging is done (subaverage size = 1)
    if include_null_case and subaverage_sizes[0] != 1:
        subaverage_sizes.insert(0, 1)

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

            for subaverage_size in subaverage_sizes:
                p = pipeline.deepcopy()

                iteration(
                    model_name=model_name,
                    training_options=training_options,
                    subaverage_size=subaverage_size,
                    pipeline_copy=p,
                    write_directory=write_directory,
                    pkl_filename_prefix=pkl_filename_prefix,
                    iteration_quantifier=subaverage_size
                )