from __future__ import annotations
from typing import *
from utils import function_label, param_labels
from EEGDataStructures import EEGSubject

ORIGINAL_SUBJECTS: List[EEGSubject] = []
SUBJECTS: List[EEGSubject] = []

@function_label("Map Class Labels")
@param_labels(["CSV Filepath"])
def GLOBAL_map_class_labels(csv_filepath: str):
    """
    Maps the raw labels for each subject in SUBJECTS as specified in the CSV. 
    """
    for subject in SUBJECTS:
        subject.map_labels(csv_filepath)

@function_label("Trim")
@param_labels(["Start Index", "End Index"])
def GLOBAL_trim_ffr(start_index: int, end_index: int):
    """
    Trims the data for all subjects in SUBJECTS
    """
    for subject in SUBJECTS:
        subject.trim(start_index=start_index, end_index=end_index)

@function_label("Subaverage Data")
@param_labels(["Subaverage Size"])
def GLOBAL_sub_average_data(size: int=5):
    """
    Subaverages the trials across all subjects
    """
    for subject in SUBJECTS:
        subject.subaverage(size=size)

@function_label("Split for Testing")
@param_labels(["Ratio"])
def GLOBAL_test_split_stratified(ratio: float=0.8):
    """
    Test-splits for all subjects.

    :param ratio: the ratio of trials to be used for training
    """
    for subject in SUBJECTS:
        subject.test_split(trials=subject.trials, ratio=ratio)

@function_label("Split into Folds")
@param_labels(["Fold Count"])
def GLOBAL_k_fold_stratified(num_folds: int=5):
    """
    EEGSubject Method Wrapper
    """
    for subject in SUBJECTS:
        subject.stratified_folds(num_folds=num_folds)

@function_label("Use GPU or CPU")
@param_labels(["Use GPU?"])
def GLOBAL_set_device(use_gpu: bool = False):
    for subject in SUBJECTS:
        subject.setDevice(use_gpu)

@function_label("Run Inference")
@param_labels([])
def GLOBAL_inference_model():
    for subject in SUBJECTS:
        subject.test_model()

@function_label("Train Model") 
@param_labels(["Model Name", "Number of Epochs", "Learning Rate", "Stopping Criteria"])
def GLOBAL_train_model(model_name: str,
                    num_epochs: int = 20,
                    lr: float = 1e-3,
                    stopping_criteria: bool = False): 
    for subject in SUBJECTS: 
        subject.train(model_name = model_name,
                    num_epochs = num_epochs, 
                    lr = lr, 
                    stopping_criteria = stopping_criteria)



#if __name__ == "__main__":
    #print(GLOBAL_load_subject_data.label)