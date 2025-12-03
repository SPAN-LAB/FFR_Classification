from pathlib import Path
import pandas as pd

from ..core import AnalysisPipeline
from ..core import get_accuracy, get_per_label_accuracy


def investigate_subaverage_size_vs_accuracy(
    subaverage_sizes: list[int],
    training_options: dict[str, any],
    subject_filepaths: list[str],
    model_names: list[str],
    output_folder_path: str
):
    def internal(model_name, subject_filepath):
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
            internal(model_name=model_name, subject_filepath=subject_filepath)
