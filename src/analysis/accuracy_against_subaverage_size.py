"""
SPAN Lab - FFR Classification

Filename: accuracy_against_subaverage_size.py
Author(s): Kevin Chen
Description: A function that evaluates a model's performance on various subaverage sizes.
"""


from pathlib import Path
import json

from .utils import get_subject_loaded_pipelines

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
    """
    This method iterates through models corresponding to the model names provided. For each model, 
    the analysis is done on each subject; for each subject, we iterate through each subaverage size
    provided. For each subaverage size, we create a JSON file containing information on the 
    predictions made for all the trials of the subject. 
    
        {
            "trials": [
                {
                    "label": "1",
                    "prediction_distribution": {
                        "1": 0.25,
                        "2": 0.3,
                        "3": 0.25,
                        "4": 0.2
                    }
                },
                {
                    "label": "2",
                    "prediction_distribution": {
                        "1": 0.05,
                        "2": 0.3,
                        "3": 0.3,
                        "4": 0.35
                    }
                },
                ...
            ]
        }
    
    This method creates a folder in the output folder path for each of the models names. 
    Then for each model, a subfolder exists for each subject it is trained on. Each subject 
    subfolder contains JSON files with the name "subaverage-<size>.json" (subaverage-1.json for 
    example). 
    """
    
    NUM_FOLDS = 5
    TRIM_START_TIME = 50
    TRIM_END_TIME = 250
    
    # Include a case where no subaveraging is done (subaverage size = 1)
    if include_null_case and subaverage_sizes[0] != 1:
        subaverage_sizes.insert(0, 1)

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
            
            for subaverage_size in subaverage_sizes: 
                p = (
                    subject_pipeline.deepcopy()
                    .trim_by_timestamp(start_time=TRIM_START_TIME, end_time=TRIM_END_TIME)
                    .subaverage(size=subaverage_size)
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
                path = Path(f"./{output_folder_path}/{model_name}/{Path(subject_filepath).stem}/subaverage-{subaverage_size}.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with path.open("w", encoding="utf-8") as file:
                    json.dump(predictions, file, indent=4)

