"""
SPAN Lab - FFR Classification

Filename: analysis.py
Author(s): Kevin Chen
Description: A template for running analyses.
"""


from src.analysis.utils import get_mats
from src.models.utils import find_models
from src.analysis import accuracy_against_subaverage_size
from src.analysis import accuracy_against_data_amount


##########################
##   GLOBAL ARGUMENTS   ##
##########################

SUBJECT_FILENAMES = ['Z:/projects/trial_classification/4tone_cell/4T1002.mat'] 
# Alternatively, use 
# SUBJECT_FILENAMES = get_mats("Z:\\projects\\trial_classification\\4tone_cell\\4T1002.mat")

MODEL_NAMES = ["LDA"] # Add more model names as needed, must correspond to classes in src/models
# Alternatively, automatically detect models with
#     `MODEL_NAMES = find_models()

TRAINING_OPTIONS = {
    "num_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.1
}

DEFER_LOADING_SUBJECTS = True


def test_subaverage():
    """
    Tests the impact of subaverage size on accuracy.
    """

    #########################
    ##   LOCAL ARGUMENTS   ##
    #########################

    SUBAVERAGE_SIZES = [2, 5, 10]
    INCLUDE_NO_SUBAVERAGING_CASE = True # Includes a case where no subaveraging is performed
    OUTPUT_FOLDER_PATH = "investigations/accuracy_against_subaverage_size"

    accuracy_against_subaverage_size(
        subaverage_sizes=SUBAVERAGE_SIZES,
        subject_filepaths=SUBJECT_FILENAMES,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_FOLDER_PATH,
        include_null_case=INCLUDE_NO_SUBAVERAGING_CASE,
        defer_subject_loading=DEFER_LOADING_SUBJECTS
    )

def test_data_amount():
    """
    Tests the impact of data amount on accuracy. The term "data amount" denotes the proportion of 
    the data that is used for training an ML model. For example, with a "data amount" of 0.5, only 
    50% of the data is used during the training of the model.
    """
    #########################
    ##   LOCAL ARGUMENTS   ##
    #########################

    MIN_TRIALS = 50
    STRIDE = 20
    OUTPUT_FOLDER_PATH = "analysis_outputs/accuracy_against_data_amount"
    
    accuracy_against_data_amount(
        min_trials=MIN_TRIALS,
        stride=STRIDE,
        subject_filepaths=SUBJECT_FILENAMES,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_FOLDER_PATH,
        defer_subject_loading=DEFER_LOADING_SUBJECTS
    )


if __name__ == "__main__":
    test_subaverage()
    # test_data_amount()
