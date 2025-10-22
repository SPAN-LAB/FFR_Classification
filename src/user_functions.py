from __future__ import annotations
from typing import *
from utils import function_label, param_labels
from EEGDataStructures import EEGSubject

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

@function_label("TODO")
@param_labels([])
def GLOBAL_inference_model():
    """
    TODO Anu
    Standalone function for inferencing on saved ONNX models.
    """
    return

@function_label("TODO")
@param_labels([])
def GLOBAL_train_model():
    """
    TODO Anu
    Standalone function for training predefined PyTorch models.
    """
    return

@function_label("Visualize Subject Per Tone")
@param_labels(["Subject Index", "Tone"])
def GLOBAL_visualize_subject_per_tone(subject_index: int=1, tone: int=1):
    """
    Visualizes the subject's data for the given tone.
    """
    SUBJECTS[subject_index - 1].visualize(label=tone)

if __name__ == "__main__":
    print(GLOBAL_load_subject_data.label)