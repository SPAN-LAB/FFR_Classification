"""
SPAN Lab - FFR Classification

Filename: demo.py
Author(s): Kevin Chen
Description: Example code for using the AnalysisPipeline APIs.
"""

from src.core import AnalysisPipeline, BlankPipeline

BASE_PATH = "/Volumes/gurindapalli/projects/trial_classification/4tone_cell/"

SUBJECT_FILEPATHS = [
    BASE_PATH + "4T1002.mat",
    BASE_PATH + "4T1004.mat",
    BASE_PATH + "4T1005.mat",
    BASE_PATH + "4T1006.mat",
    BASE_PATH + "4T1007.mat",
    BASE_PATH + "4T1008.mat",
    BASE_PATH + "4T1009.mat",
    BASE_PATH + "4T1010.mat",
    BASE_PATH + "4T1012.mat",
    BASE_PATH + "4T1014.mat",
    BASE_PATH + "4T1015.mat",
]

loading_result = BlankPipeline()
trimming_result = BlankPipeline()
subaverage_and_fold_result = BlankPipeline()

p = (
    AnalysisPipeline()
    .load_subjects(SUBJECT_FILEPATHS)
    .trim_by_timestamp(start_time=50, end_time=250)
    .subaverage(5)
    .extract_features(["pitchtrack", "autocorr"], concatenate=False)
    .fold(5)
    .evaluate_model(
        model_name="TransformerMulti",
        training_options={
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.1,
            "patience": 20,
            "min_delta": 0.001,
            "patch_size": 8,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "max_tokens": 2048,
        }
    )
)