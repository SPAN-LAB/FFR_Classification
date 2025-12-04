from pathlib import Path
import pandas as pd
from copy import deepcopy
from random import sample

from ..core import AnalysisPipeline, PipelineState
from ..core import get_accuracy, get_per_label_accuracy

from .utils import get_subject_loaded_pipelines

from ..core.plots import plot_confusion_matrix, plot_roc_curve


def accuracy_against_data_amount(
    min_trials: int, 
    stride: int,
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
):
    # Constants
    SUBAVERAGE_SIZE = 1
    NUM_FOLDS = 5
    TRIM_START_TIME = 50
    TRIM_END_TIME = 250

    # Cached subject-loaded pipeline states
    subject_loaded_pipelines = get_subject_loaded_pipelines(subject_filepaths)

    def internal(subject_filepath, model_name):
        results = []
        labels = None
        headers = ["Data Amount", "Accuracy"]

        data_amount = min_trials
        while data_amount < len(subject_loaded_pipelines[subject_filepath].subjects[0].trials):
            base = subject_loaded_pipelines[subject_filepath]
            p0 = PipelineState()
            base.save(to=p0)
            
            p1 = (
                p0
                .trim_by_timestamp(start_time=TRIM_START_TIME, end_time=TRIM_END_TIME)
            )

            subject = p1.subjects[0]
            trials = sample(deepcopy(subject.trials), data_amount)
            for i, trial in enumerate(trials):
                # Ensure that the trial's index matches its `trial_index` attribute
                trial.trial_index = i
            subject.trials = trials

            _ = (
                p1
                .subaverage(size=SUBAVERAGE_SIZE)
                .fold(num_folds=NUM_FOLDS)
                .evaluate_model(
                    model_name=model_name,
                    training_options=training_options
                )
            )
            if subject.trials[0].prediction is None:
                continue
            
            plot_confusion_matrix(subject=subject, filepath=f"{output_folder_path}/{model_name}/{subject.name}/confusion/{data_amount}.svg")
            plot_roc_curve(subject=subject, filepath=f"{output_folder_path}/{model_name}/{subject.name}/roc/{data_amount}.svg")

            if labels is None:
                labels = subject.labels_map.keys()
                for label in labels:
                    headers.append(f"Accuracy (label={label})")
            
            row_data = [data_amount, get_accuracy(subject)]
            per_label_accuracies = get_per_label_accuracy(subject)
            for label in labels:
                row_data.append(per_label_accuracies[label])
            
            results.append(row_data)
            
            data_amount += stride
        
        # Save the results
        output_filepath = Path(output_folder_path) / model_name
        output_filepath.mkdir(parents=True, exist_ok=True)
        output_filepath = output_filepath / f"{Path(subject_filepath).stem}.csv"

        df = pd.DataFrame(results, columns=headers)
        df.to_csv(output_filepath, index=False)

        print(f"Results saved to: {output_filepath}")
        return results

    for model_name in model_names:
        for subject_filepath in subject_filepaths:
            internal(subject_filepath=subject_filepath, model_name=model_name)