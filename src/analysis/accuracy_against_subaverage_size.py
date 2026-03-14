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
from ..time import TimeKeeper
from datetime import datetime, timezone
from ..printing.logging import log, is_empty


def accuracy_against_subaverage_size(
    subaverage_sizes: list[int],
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    include_null_case: bool = True,
    defer_subject_loading: bool = True
):
    
    # MARK: Setting up variables
    
    # Name used for this investigation's independent variable
    independent_var_name = "subaverage-size"
    
    # The prefix in the filename of each subject that is saved as a pickle file
    pkl_filename_prefix = f"{independent_var_name}-"
    
    # Stands for "per-subject time log filename"; 
    # The filename for the file to which the time spent evaluating a model 
    # trained on EACH subject is logged.
    pst_log_filename = "times-log.csv"
    
    # Stands for "per-iteation time log filename"
    pit_log_filename = "times-log.csv"
    
    # Stands for "per-model time log filename"
    pmt_log_filename = "times-log.csv"
    
    pstl_description_filename = "times-log-note.txt"
        
    # Include a case where no subaveraging is done (subaverage size = 1)
    if include_null_case and subaverage_sizes[0] != 1:
        subaverage_sizes.insert(0, 1)

    subject_loaded_pipelines = None
    if not defer_subject_loading:
        subject_loaded_pipelines = get_subject_loaded_pipelines(subject_filepaths)
    
    per_model_tk = TimeKeeper()
    per_subject_tk = TimeKeeper()
    per_iteration_tk = TimeKeeper()
    per_model_tk.reset()
    per_subject_tk.reset()
    per_iteration_tk.reset()

    # Header for the per-model times CSV
    pmt_log_directory = Path(output_folder_path)
    pmt_log_filepath = pmt_log_directory / pmt_log_filename
    if not pmt_log_filepath.exists() or is_empty(pmt_log_filepath):
        log("model-name,duration(seconds),subjects,completion-time(utc)", pmt_log_filepath)

    for model_name in model_names:
        
        # Header for the CSV
        pst_log_directory = Path(output_folder_path) / model_name
        pst_log_filepath = pst_log_directory / pst_log_filename
        if not pst_log_filepath.exists() or is_empty(pst_log_filepath):
            log("subject-identifier,duration(seconds),completion-time(utc)", pst_log_filepath)
        
        # Create a guide to the time log
        pstl_description_filepath = pst_log_directory / pstl_description_filename
        if not pstl_description_filepath.exists() or is_empty(pstl_description_filepath):
            pstl_description = (
                "Each value in the \"duration\" column in times-log.csv "
                "indicates the total time elapsed during the "
                "evaluation of the model on each subject."
            )
            log(pstl_description, pstl_description_filepath)
        
        per_model_tk.reset()
        per_model_tk.start()
        
        for subject_filepath in subject_filepaths:
            
            # The base subject pipeline state used for this subject. 
            # Do not modify, only deeply copy.
            if not defer_subject_loading: 
                pipeline = subject_loaded_pipelines[subject_filepath]
            else:
                pipeline = AnalysisPipeline().load_subjects(subject_filepath)

            subject_filename = Path(subject_filepath).stem
            write_directory = Path(output_folder_path) / model_name / subject_filename
            
            per_subject_tk.start()
            
            for subaverage_size in subaverage_sizes: 
                iteration(
                    model_name,
                    training_options,
                    subaverage_size,
                    pipeline.deepcopy(),
                    write_directory,
                    pkl_filename_prefix,
                    subaverage_size,
                    independent_var_name,
                    per_iteration_tk,
                    pit_log_filename
                )

            per_subject_tk.stop()
            log(
                (
                    f"{subject_filename}"
                    + f",{per_subject_tk.accumulated_duration}"
                    + f",{datetime.now(timezone.utc).isoformat()}"
                ),
                pst_log_filepath
            )
        
        per_model_tk.stop()
        
        subject_identifiers = [
            Path(subject_filepath).stem 
            for subject_filepath in subject_filepaths
        ]
        log(
            (
                f"{model_name}"
                + f",{per_model_tk.accumulated_duration}"
                + "," + "|".join(subject_identifiers)
                + f",{datetime.now(timezone.utc).isoformat()}"
            ),
            pmt_log_filepath
        )