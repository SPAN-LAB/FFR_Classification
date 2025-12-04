from src.analysis.utils import get_mats
from src.models.utils import find_models
from src.analysis import accuracy_against_subaverage_size
from src.analysis import accuracy_against_data_amount


SUBJECT_FILENAMES = ["REPLACE ME IF YOU AGREE MAC IS THE BEST"]
# Alternatively, use 
#     `SUBJECT_FILENAMES = get_mats("PATH TO FOLDER CONTAINING MAT FILES")`
MODEL_NAMES = ["FFNN"]
# Alternatively, automatically detect models with
#     `MODEL_NAMES = find_models()
TRAINING_OPTIONS = {
    "num_epochs": 20,
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.1
}

def test_subaverage():
    # These are constants specific to the subaveraging investigation
    SUBAVERAGE_SIZES = [2, 5, 10, 20, 40, 80]
    INCLUDE_NO_SUBAVERAGING_CASE = True # Includes a case where no subaveraging is performed
    OUTPUT_FOLDER_PATH = "analysis_output/accuracy_against_subaverage_size"

    accuracy_against_subaverage_size(
        subaverage_sizes=SUBAVERAGE_SIZES,
        subject_filepaths=SUBJECT_FILENAMES,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_FOLDER_PATH,
        include_null_case=INCLUDE_NO_SUBAVERAGING_CASE
    )

def test_data_amount():
    # These are constants specific to the subaveraging investigation
    MIN_TRIALS = 50
    STRIDE = 20
    OUTPUT_FOLDER_PATH = "analysis_outputs/accuracy_against_data_amount"
    
    accuracy_against_data_amount(
        min_trials=MIN_TRIALS,
        stride=STRIDE,
        subject_filepaths=SUBJECT_FILENAMES,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_FOLDER_PATH
    )

if __name__ == "__main__":
    test_subaverage()