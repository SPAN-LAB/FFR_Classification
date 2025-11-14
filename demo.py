from src.core import AnalysisPipeline

from local.constants import ALL_PATH
# Replace this with a variable ``ALL_PATH``, a string representing the path to the directory
# containing the data. Example:
# 
#     ALL_PATH = "area51/martian_subject_42/eeg_data"
#
# Using the ``ALL_PATH`` variable is NOT required. If you use a different variable name,
# update the variable name in the ``.load_subjects(...)`` line.


p = (
    AnalysisPipeline()
    .load_subjects(ALL_PATH)
    .trim_by_timestamp(start_time=0, end_time=float("inf")) # Keep all starting from 0 ms
    .subaverage()
    .fold()
    .evaluate_model(
        model_name="FFNN",
        training_options={
            "num_epochs": 20,
            "batch_size": 64,
            "learning_rate": 0.001
        }    
    )
)