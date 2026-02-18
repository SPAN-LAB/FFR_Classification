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

ALL_PATH = ["/Volumes/gurindapalli/projects/trial_classification/4tone_cell/4T1005.mat"]

p = (
    AnalysisPipeline()
    .load_subjects(ALL_PATH)
    .save(to=loading_result)
    .trim_by_timestamp(start_time=0, end_time=float("inf")) # Keep all starting from 0 ms
    .save(to=trimming_result)
    .subaverage(30)
    .fold(5)
    .save(to=subaverage_and_fold_result)
    .evaluate_model(
        model_name="HMM",
        training_options={
            "n_states": 5,
            "n_iter": 90,
            "tol": 1e-3,
            "covariance_type": "diag",
            "min_covar": 1e-3,
            "random_state": 42,
            "signal_attr": "data",
            "feature_mode": "raw_delta_delta2",
            "normalize_features": True,
            "per_sequence_zscore": True,
            "temporal_downsample": 4,
            "max_sequence_length": 1400,
            "n_restarts": 6,
            "max_fit_calls_per_class": 10,
            "use_class_priors": True,
            "score_normalization": "length",
            "auto_state_cap_by_samples": True,
            "min_sequences_per_state": 3,
            "state_selection_criterion": "none",
            "hybrid_centroid_weight": 0.1,
            "default_eval_folds": 5,
            "verbose": True
        }    
    )
)
