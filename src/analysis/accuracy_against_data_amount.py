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
from ..configurations import SUBAVERAGE_SIZE
from .iteration import iteration
from ..time import TimeKeeper
from datetime import datetime, timezone
from ..printing.logging import log, is_empty

def accuracy_against_data_amount(
    min_trials: int, 
    stride: int,
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    defer_subject_loading: bool = True
):
    
    # MARK: Setting up variables
    
    # Name used for this investigation's independent variable
    independent_var_name = "data-amount"
    
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

    # If don't defer subject loading, load all the subjects now
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
                    data_amount,
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
