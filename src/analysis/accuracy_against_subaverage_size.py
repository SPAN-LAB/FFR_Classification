from pathlib import Path
import pandas as pd

from ..core import AnalysisPipeline
from ..core import get_accuracy, get_per_label_accuracy
from ..core.plots import plot_confusion_matrix, plot_roc_curve

def accuarcy_against_subaverage_size(
    subaverage_sizes: list[int],
    subject_filepaths: list[str],
    model_names: list[str],
    training_options: dict[str, any],
    output_folder_path: str,
    include_null_case: bool = True,
):
    # Include a case where no subaveraging is done (subaverage size = 1)
    if include_null_case and subaverage_sizes[0] != 1:
        subaverage_sizes.insert(index=0, object=1)

    def internal(subject_filepath, model_name):
        """
        Performs the analysis on a single model and single subject
        """
        results = []
        labels = None
        headers = ["Subaverage Size", "Accuracy"]

        for subaverage_size in subaverage_sizes:
            p = (
                AnalysisPipeline()
                .load_subjects(subject_filepath)
                .subaverage(size=subaverage_size)
                .fold(num_folds=5)
                .evaluate_model(
                    model_name=model_name,
                    training_options=training_options
                )
            )

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
