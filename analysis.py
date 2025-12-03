from src.analysis import investigate_subaverage_size_vs_accuracy
from src.analysis.utils import get_mats


investigate_subaverage_size_vs_accuracy(
    subaverage_sizes=list(range(2, 25, 2)), # You can use [1, 5, 10, 50], for example
    subject_filepaths=[get_mats("< ...enter your filepath here... >")],
    model_names=["FFNN", "CNN"],
    training_options={
        "num_epochs": 20,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.1
    },
    output_folder_path="analysis_outputs"
)
