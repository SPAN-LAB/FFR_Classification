from EEGDataStructures import EEGSubject

def GLOBAL_map_class_labels(e: EEGSubject, csv_filepath: str):
    """
    TODO Kevin
    EEGTrial Method Wrapper
    """
    e.map_labels(csv_filepath)

def GLOBAL_trim_ffr(e: EEGSubject, start_index: int, end_index: int):
    """
    TODO Kevin
    EEGTrial Method Wrapper
    """
    e.trim(start_index=start_index, end_index=end_index)

def GLOABL_sub_average_data(e: EEGSubject, size: int):
    """
    TODO Kevin
    EEGSubject Method Wrapper
    """
    e.subaverage(size=size)

def GLOBAL_test_split_stratified(e: EEGSubject, ratio: float):
    """
    TODO Kevin
    EEGSubject Method Wrapper
    """
    e.test_split(trials=e.trials, ratio=ratio)

def GLOBAL_k_fold_stratified(e: EEGSubject, num_folds: int):
    """
    TODO Kevin
    EEGSubject Method Wrapper
    """
    e.stratified_folds(num_folds=num_folds)

def GLOBAL_inference_model():
    """
    TODO Anu
    Standalone function for inferencing on saved ONNX models.
    """
    return

def GLOBAL_train_model():
    """
    TODO Anu
    Standalone function for training predefined PyTorch models.
    """
    return