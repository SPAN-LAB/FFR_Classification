"""
SPAN Lab - FFR Classification

Filename: demo.py
Author(s): Kevin Chen
Description: Example code for using the AnalysisPipeline APIs.
"""


from src.core import AnalysisPipeline, BlankPipeline

# Replace this with a string variable representing the path to the directory containing the data. 
# Example:
# 
#     PATH = "area51/martian_subject_42/eeg_data"
#
# Swap this variable into the ``load_subjects`` method on line 18 below.

SUBJECT_FILEPATHS = ["4T1002.mat", "4T1004.mat","4T1005.mat","4T1006.mat","4T1007.mat","4T1008.mat","4T1009.mat","4T1010.mat","4T1012.mat","4T1014.mat","4T1015.mat"]

loading_result = BlankPipeline()
trimming_result = BlankPipeline()
subaverage_and_fold_result = BlankPipeline()

p = (
    AnalysisPipeline()
    .load_subjects(SUBJECT_FILEPATHS[0])
    .save(to=loading_result)
    .trim_by_timestamp(start_time=0, end_time=float("inf")) # Keep all starting from 0 ms
    .save(to=trimming_result)
    .subaverage(5)
    .fold(5)
    .save(to=subaverage_and_fold_result)
    .evaluate_model(
        model_name="RNN",
        training_options={
            "num_epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.1
        }    
    )
)
