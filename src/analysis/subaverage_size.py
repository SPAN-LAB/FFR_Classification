from src.analysis import accuracy_against_subaverage_size


def analyze(model_name: str, subject_filepath: str, output_dirpath: str):

    MODEL_NAMES = [model_name]
    SUBJECT_FILEPATHS = [subject_filepath]
    OUTPUT_DIR_PATH = output_dirpath
    SUBAVERAGE_SIZES = [i for i in range(5, 125 + 1, 5)]
    INCLUDE_NULL_CASE = True
    DEFER_SUBJECT_LOADING = False
    TRAINING_OPTIONS = {
        "num_epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1
    }

    accuracy_against_subaverage_size(
        subaverage_sizes=SUBAVERAGE_SIZES,
        subject_filepaths=SUBJECT_FILEPATHS,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_DIR_PATH,
        include_null_case=INCLUDE_NULL_CASE,
        defer_subject_loading=DEFER_SUBJECT_LOADING
    )
