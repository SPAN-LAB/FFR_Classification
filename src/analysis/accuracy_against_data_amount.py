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
from random import sample

from ..core import AnalysisPipeline

from .utils import get_subject_loaded_pipelines


def accuracy_against_data_amount(
    min_trials: int, 
    stride: int,
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    defer_subject_loading: bool = True
):
    SUBAVERAGE_SIZE = 1
    NUM_FOLDS = 5
    TRIM_START_TIME = 50
    TRIM_END_TIME = 250

    subject_loaded_pipelines = None
    if not defer_subject_loading:
        subject_loaded_pipelines = get_subject_loaded_pipelines(subject_filepaths)

    for model_name in model_names:
        for subject_filepath in subject_filepaths:
            
            # The base subject pipeline state used for this subject; do not modify, only deeply copy
            if not defer_subject_loading: 
                subject_pipeline = subject_loaded_pipelines[subject_filepath]
            else:
                subject_pipeline = AnalysisPipeline().load_subjects(subject_filepath)
            
            data_amount = min_trials
            while data_amount <= len(subject_pipeline.subjects[0].trials):
                # Keep only `data_amount` number of trials
                reduced_trials = sample(
                    deepcopy(subject_pipeline.deepcopy().subjects[0].trials), 
                    data_amount
                )
                
                # Index them properly
                for i, trial in enumerate(reduced_trials):
                    trial.trial_index = i
                
                reduced_starting_pipeline = subject_pipeline.deepcopy()
                reduced_starting_pipeline.subjects[0].trials = reduced_trials
                
                p = (
                    reduced_starting_pipeline
                    .trim_by_timestamp(start_time=TRIM_START_TIME, end_time=TRIM_END_TIME)
                    .subaverage(size=SUBAVERAGE_SIZE)
                    .fold(num_folds=NUM_FOLDS)
                    .evaluate_model(
                        model_name=model_name,
                        training_options=training_options
                    )
                )
                subject = p.subjects[0]
                
                # Create the dictionary
                predictions = {"trials": []}
                for trial in subject.trials:
                    predictions["trials"].append({
                        "label": trial.label,
                        "prediction_distribution": trial.prediction_distribution
                    })
                
                # Save predictions to <output_dir_path>/<model-name>/<subject-name>/subaverage-<size>.json
                path = Path(f"./{output_folder_path}/{model_name}/{Path(subject_filepath).stem}/data-amount-{data_amount}.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Remove training data from subject for smaller file size
                for trial in subject.trials:
                    trial.data = []
                    trial.timestamps = []
                subject.folds = []
                
                with path.open("wb") as file:
                    pickle.dump(subject, file)
                    
                data_amount += stride

