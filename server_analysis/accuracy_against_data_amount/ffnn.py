from src.analysis import accuracy_against_data_amount
from src.analysis.utils import get_mats
from local.constants import *


# ------------------------------------------- Arguments -------------------------------------------

MIN_TRIALS = 3000

STRIDE = 100 # Trial number is incremented by this amount

# SUBJECT_FILEPATHS = get_mats(ALL_PATH)
SUBJECT_FILEPATHS = [GOOD_D_PATH]

MODEL_NAMES = ["FFNN"]

TRAINING_OPTIONS = {
    "num_epochs": 20,
    "batch_size": 10000,
    "learning_rate": 0.001,
    "weight_decay": 0.1
}    

OUTPUT_DIR_PATH = "analysis-results/data-amount"

DEFER_SUBJECT_LOADING = True

# --------------------------------------- Perform Analysis ----------------------------------------

accuracy_against_data_amount(
    min_trials=MIN_TRIALS,
    stride=STRIDE,
    subject_filepaths=SUBJECT_FILEPATHS,
    model_names=MODEL_NAMES,
    training_options=TRAINING_OPTIONS,
    output_folder_path=OUTPUT_DIR_PATH,
    defer_subject_loading=DEFER_SUBJECT_LOADING
)
