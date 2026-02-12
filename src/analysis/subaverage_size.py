from src.analysis import accuracy_against_subaverage_size
from .config import SUBJECT_FILEPATHS


def analyze(model_name: str):
    
    # -------------------------- TODO: Change these! --------------------------
    
    # IMPORTANT: Ensure that this path is different 
    # from the one used in `/src/analysis/data_amount.py`.
    OUTPUT_DIR_PATH = "analyses/subaverage-analysis" 
    
    # -------------------------------------------------------------------------
    
    SUBAVERAGE_SIZES = [i for i in range(5, 125 + 1, 5)]
    MODEL_NAMES = [model_name]
    INCLUDE_NULL_CASE = True
    DEFER_SUBJECT_LOADING = True
    TRAINING_OPTIONS = {
        "num_epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1
    }
    
    # Check SUBJECT_FILEPATHS
    if SUBJECT_FILEPATHS is None or SUBJECT_FILEPATHS == []:
        print("Error: SUBJECT_FILEPATHS is empty.")
        print("       You may have forgotten to configure it.")
        print("       The configuration file is in /src/analysis/config.py")
        return
    
    accuracy_against_subaverage_size(
        subaverage_sizes=SUBAVERAGE_SIZES,
        subject_filepaths=SUBJECT_FILEPATHS,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_DIR_PATH,
        include_null_case=INCLUDE_NULL_CASE,
        defer_subject_loading=DEFER_SUBJECT_LOADING
    )