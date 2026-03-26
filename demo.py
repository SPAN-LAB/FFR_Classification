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
#from local.constants import *


loading_result = BlankPipeline()
trimming_result = BlankPipeline()
subaverage_and_fold_result = BlankPipeline()

ALL_PATH = ["/Volumes/gurindapalli/projects/trial_classification/4tone_cell/4T1015.mat"]

p = (
    AnalysisPipeline()
    .load_subjects(ALL_PATH)
    .save(to=loading_result)
    .trim_by_timestamp(start_time=0, end_time=float("inf")) # Keep all starting from 0 ms
    .save(to=trimming_result)
    .subaverage(100)
    .fold(5)
    .save(to=subaverage_and_fold_result)
    .evaluate_model(
        model_name="HMM",
        training_options={
            "n_states": 3,
            "n_iter": 30,
            "fs": 16384.0,
        }
    )
)
