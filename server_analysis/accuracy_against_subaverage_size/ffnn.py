from src.analysis import accuracy_against_subaverage_size
from local.constants import *

# ------------------------------------------- Arguments -------------------------------------------

SUBAVERAGE_SIZES = [2, 4, 8, 16, 32, 64, 128]

SUBJECT_FILEPATHS = [GOOD_D_PATH] # TODO

MODEL_NAMES = ["FFNN"]

TRAINING_OPTIONS = {
    "num_epochs": 20,
    "batch_size": 10000,
    "learning_rate": 0.001,
    "weight_decay": 0.1
}    

OUTPUT_DIR_PATH = "analysis-results/subaverage-size" # TODO 

INCLUDE_NULL_CASE = True

DEFER_SUBJECT_LOADING = True

# --------------------------------------- Perform Analysis ----------------------------------------

accuracy_against_subaverage_size(
    subaverage_sizes=SUBAVERAGE_SIZES,
    subject_filepaths=SUBJECT_FILEPATHS,
    model_names=MODEL_NAMES,
    training_options=TRAINING_OPTIONS,
    output_folder_path=OUTPUT_DIR_PATH,
    include_null_case=INCLUDE_NULL_CASE,
    defer_subject_loading=DEFER_SUBJECT_LOADING
)