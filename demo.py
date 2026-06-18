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
    .trim_by_timestamp(50, 250)
    .subaverage(5)
    .extract_features(["autoencoder_latent"])
    .fold(5)
    .evaluate_model("FFNN", training_options={
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "patience": 50,
        "min_delta": 0.001,
    })
)