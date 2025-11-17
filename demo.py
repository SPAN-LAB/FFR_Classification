from src.core import AnalysisPipeline, PipelineState
from src.core import EEGSubject

from local.constants import *
# Replace this with a variable ``ALL_PATH``, a string representing the path to the directory
# containing the data. Example:
# 
#     ALL_PATH = "area51/martian_subject_42/eeg_data"
#
# Using the ``ALL_PATH`` variable is NOT required. If you use a different variable name,
# update the variable name in the ``.load_subjects(...)`` line.

loading_result = PipelineState()
trimming_result = PipelineState()
subaverage_and_fold_result = PipelineState()

p = (
    AnalysisPipeline()
    .load_subjects(filepath_list=GOOD_D_PATH)
    .save(to=loading_result)
    .trim_by_timestamp(start_time=0, end_time=float("inf")) # Keep all starting from 0 ms
    .save(to=trimming_result)
    .subaverage(20)
    .fold()
    .save(to=subaverage_and_fold_result)
    .evaluate_model(
        model_name="FFNN",
        training_options={
            "num_epochs": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.1
        }    
    )
)