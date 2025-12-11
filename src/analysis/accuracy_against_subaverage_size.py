from pathlib import Path
import pandas as pd

from .utils import get_subject_loaded_pipelines, save_times
from ..time import TimeKeeper

from ..core import AnalysisPipeline, PipelineState
from ..core import get_accuracy, get_per_label_accuracy
from ..core.plots import plot_confusion_matrix, plot_roc_curve

def accuracy_against_subaverage_size(
    subaverage_sizes: list[int],
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    include_null_case: bool = True,
):
    # Include a case where no subaveraging is done (subaverage size = 1)
    if include_null_case and subaverage_sizes[0] != 1:
        subaverage_sizes.insert(0, 1)

    # Cached subject-loaded pipeline states
    subject_loaded_pipelines = get_subject_loaded_pipelines(subject_filepaths)

    def internal(subject_filepath, model_name):
        """
        Performs the analysis on a single model and single subject
        """
        results = []
        labels = None
        headers = ["Subaverage Size", "Accuracy"]
        time_keeper = TimeKeeper()
        durations = [] # The duration elapsed for each iteration in the for-loop below

        for subaverage_size in subaverage_sizes:
            test = PipelineState()
            p = (
                subject_loaded_pipelines[subject_filepath].deepcopy()
                .subaverage(size=subaverage_size)
                .save(to=test)
                .fold(num_folds=5)
                .evaluate_model(
                    model_name=model_name,
                    training_options=training_options
                )
            )
            # print(f"{len(p.subjects) = }")
            # print(f"{len(test.subjects) = }")
            # print(f"{len(p.subjects[0].trials) = }")
            
            # print(f"On this iteration, there were {len(test.subjects[0].trials)} trials after subaveraging")

            subject = p.subjects[0]
            
            plot_confusion_matrix(subject=subject, filepath=f"{output_folder_path}/{model_name}/{subject.name}/confusion/{subaverage_size}.svg")
            plot_roc_curve(subject=subject, filepath=f"{output_folder_path}/{model_name}/{subject.name}/roc/{subaverage_size}.svg")

            # Ensure that labels and headers are arranged consistently for all subaverage sizes
            if labels is None:
                labels = subject.labels_map.keys()
                for label in labels:
                    headers.append(f"Accuracy (label={label})")

            row_data = [subaverage_size, get_accuracy(subject)]
            per_label_accuracies = get_per_label_accuracy(subject)
            for label in labels:
                row_data.append(per_label_accuracies[label])

            results.append(row_data)
            t = time_keeper.lap_time()
            durations.append(t)
            print(f"{(t):.4f}s elapsed for size = {subaverage_size}")

        # Save the results
        output_filepath = Path(output_folder_path) / model_name
        output_filepath.mkdir(parents=True, exist_ok=True)
        end = time_keeper.end_time()
        _subaverage_sizes = ["Subaverage Size"] + subaverage_sizes + ["Total"]
        _times = ["Time"] + durations + [end]
        
        save_times(
            _subaverage_sizes, 
            _times, 
            output_filepath / f"{Path(subject_filepath).stem}.txt"
        )
        output_filepath = output_filepath / f"{Path(subject_filepath).stem}.csv"
        
        # Save the times
        # save_times(subaverage_sizes, durations, output_filepath / f"{Path(subject_filepath).stem}.txt")
        

        df = pd.DataFrame(results, columns=headers)
        df.to_csv(output_filepath, index=False)
        print(f"{(end):.4f}s elapsed in total; results saved to: {output_filepath}")
        return results

    for model_name in model_names:
        for subject_filepath in subject_filepaths:
            internal(subject_filepath=subject_filepath, model_name=model_name)
