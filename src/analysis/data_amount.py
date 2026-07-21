from src.analysis import accuracy_against_data_amount


def analyze(model_name: str, subject_filepath: str, output_dirpath: str):

    MODEL_NAMES = [model_name]
    SUBJECT_FILEPATHS = [subject_filepath]
    OUTPUT_DIR_PATH = output_dirpath
    MIN_TRIALS = 100
    STRIDE = 100 # Trial number is incremented by this amount
    DEFER_SUBJECT_LOADING = False
    TRAINING_OPTIONS = {
        "num_epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1
    }

    accuracy_against_data_amount(
        min_trials=MIN_TRIALS,
        stride=STRIDE,
        subject_filepaths=SUBJECT_FILEPATHS,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_DIR_PATH,
        defer_subject_loading=DEFER_SUBJECT_LOADING
    )
