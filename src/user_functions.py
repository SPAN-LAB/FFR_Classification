from __future__ import annotations
from typing import *
from utils import function_label, param_labels, options_provided
from Option import Option, ComputableOption
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

@function_label("Visualize Grand Average")
@param_labels(["Grouping Method", "Number"])
@options_provided([Option("Stimulus", "Class"), None])
def GLOBAL_visualize_grand_average(grouping_method: str="Stimulus"):
    """
    TODO
    """
    # Create a pseudo-subject that is the amalgamation of all subjects 
    p_subject = EEGSubject.pseudo_subject(SUBJECTS)


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

@function_label("Select Hardware")
@param_labels(["Use GPU?"])
def GLOBAL_set_device(use_gpu: bool = False):
    for subject in SUBJECTS:
        subject.setDevice(use_gpu)

@function_label("TODO")
@param_labels([])
def GLOBAL_inference_model():
    """
    TODO Anu
    Standalone function for inferencing on saved ONNX models.
    """
    return

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

