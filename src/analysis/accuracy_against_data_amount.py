from pathlib import Path
import pandas as pd
from copy import deepcopy
from random import sample

from ..core import AnalysisPipeline
from ..core import get_accuracy, get_per_label_accuracy

from .utils import get_subject_loaded_pipelines
from .utils import save_times
from ..time import TimeKeeper

from ..core.plots import plot_confusion_matrix, plot_roc_curve


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
            
            data_amounts = []
            headers = ["Data Amount", "Accuracy"]
            accuracies = []
            labels = None
            time_keeper = TimeKeeper()
            durations = []
            
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
                
                # Ensure that labels and headers are arranged consistently for all subaverage sizes
                # This is run only once
                if labels is None:
                    labels = subject.labels_map.keys()
                    for label in labels:
                        headers.append(f"Accuracy (label={label})")
                
                # Format the data for this iteration
                row_data = [data_amount, get_accuracy(subject)]
                per_label_accuracies = get_per_label_accuracy(subject)
                for label in labels:
                    row_data.append(per_label_accuracies[label])
                    
                accuracies.append(row_data)
                t = time_keeper.lap_time()
                durations.append(t)
                print(f"{(t):.4f}s elapsed for {data_amount = }")
                    
                plot_confusion_matrix(
                    subject=subject, 
                    filepath=f"{output_folder_path}/{model_name}/{subject.name}/confusion/{data_amount}.svg"
                )
                plot_roc_curve(
                    subject=subject, 
                    filepath=f"{output_folder_path}/{model_name}/{subject.name}/roc/{data_amount}.svg"
                )
                
                data_amounts.append(data_amount)
                data_amount += stride
            
            # Save the results
            output_filepath = Path(output_folder_path) / model_name
            output_filepath.mkdir(parents=True, exist_ok=True)
            end = time_keeper.end_time()
            _data_amounts = ["Data Amount"] + data_amounts + ["Total"]
            _times = ["Time"] + durations + [end]
            
            save_times(
                _data_amounts, 
                _times, 
                output_filepath / f"{Path(subject_filepath).stem}.txt"
            )
            output_filepath = output_filepath / f"{Path(subject_filepath).stem}.csv"
    
            df = pd.DataFrame(accuracies, columns=headers)
            df.to_csv(output_filepath, index=False)
            print(f"{(end):.4f}s elapsed in total; results saved to: {output_filepath}")

