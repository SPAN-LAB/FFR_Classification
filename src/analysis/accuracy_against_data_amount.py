"""
SPAN Lab - FFR Classification

Filename: accuracy_against_data_amount.py
Author(s): Kevin Chen
Description: A function that evaluates a model's performance on various data amounts.
    Data amount refers to the number of trials whose data is used for training.
"""


from pathlib import Path
import pickle
from copy import deepcopy

from .utils import get_subject_loaded_pipelines, stratified_deterministic_sample, strip_data_away
from ..core import AnalysisPipeline
from ..configurations import SUBAVERAGE_SIZE, NUM_FOLDS, TRIM_START_TIME, TRIM_END_TIME

def accuracy_against_data_amount(
    min_trials: int, 
    stride: int,
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    defer_subject_loading: bool = True
):
    
    pkl_filename_prefix = "data-amount-"

    # If don't defer subject loading, load all the subjects now
    subject_loaded_pipelines = None
    if not defer_subject_loading:
        subject_loaded_pipelines = get_subject_loaded_pipelines(subject_filepaths)
    
    def iteration(model_name, subject_filepath, write_directory):
        
        # The base subject pipeline state used for this subject. 
        # Do not modify, only deeply copy.
        if not defer_subject_loading: 
            subject_pipeline = subject_loaded_pipelines[subject_filepath]
        else:
            subject_pipeline = AnalysisPipeline() \
                .load_subjects(subject_filepath)
        
        # Determine the maximum data_amount value
        max_data_amount = len(subject_pipeline.subjects[0].trials)
        r = (max_data_amount - min_trials) % stride
        if r != 0:
            max_data_amount -= r
            
        for data_amount in range(min_trials, max_data_amount, stride):
            
            # Create the reduced pipeline
            reduced_pipeline = deepcopy(subject_pipeline)
            trials = stratified_deterministic_sample(
                reduced_pipeline.subjects[0], data_amount)
            reduced_pipeline.subjects[0].trials = trials
            
            p = (
                reduced_pipeline
                .trim_by_timestamp(start_time=TRIM_START_TIME, end_time=TRIM_END_TIME)
                .subaverage(size=SUBAVERAGE_SIZE)
                .fold(num_folds=NUM_FOLDS)
                .evaluate_model(
                    model_name=model_name,
                    training_options=training_options
                )
            )
            
            subject = p.subjects[0]
            # So that storage size is smaller, since we only need predictions
            strip_data_away(subject)
            
            # Create needed folders and save
            full = write_directory / f"{pkl_filename_prefix}{data_amount}.pkl"
            full.parent.mkdir(parents=True, exist_ok=True)
            with full.open("wb") as file:
                pickle.dump(subject, file)

    for model_name in model_names:
        for subject_filepath in subject_filepaths:
            subject_filename = Path(subject_filepath).stem
            write_directory = Path(output_folder_path) / model_name / subject_filename
            iteration(model_name, subject_filepath, write_directory)
