from __future__ import annotations
from typing import *
from .utils import function_label, param_labels
from ..core.EEGSubject import EEGSubject

ORIGINAL_SUBJECTS: List[EEGSubject] = []
SUBJECTS: List[EEGSubject] = []

@function_label("Map Class Labels")
@param_labels(["CSV Filepath"])
def GLOBAL_map_class_labels(csv_filepath: str):
    """
    Maps raw labels to mapped labels for each subject according to CSV mapping.
    """
    for subject in SUBJECTS:
        subject.map_trial_labels(csv_filepath)

@function_label("Trim by Index")
@param_labels(["Start Index", "End Index"])
def GLOBAL_trim_by_index(start_index: int, end_index: int):
    """
    Trims data for all subjects using sample indices (inclusive).
    """
    for subject in SUBJECTS:
        subject.trim_by_index(start_index=start_index, end_index=end_index)

@function_label("Trim by Time")
@param_labels(["Start Time", "End Time"])
def GLOBAL_trim_by_time(start_time: float, end_time: float):
    for subject in SUBJECTS:
        subject.trim_by_timestamp(start_time=start_time, end_time=end_time)

@function_label("Subaverage Data")
@param_labels(["Subaverage Size"])
def GLOBAL_sub_average_data(size: int=5):
    for subject in SUBJECTS:
        subject.subaverage(size=size)

@function_label("Split into Folds")
@param_labels(["Fold Count"])
def GLOBAL_split_into_folds(num_folds: int=5):
    for subject in SUBJECTS:
        subject.fold(num_folds=num_folds)

###############################################################################


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

@function_label("Train with Multiple Subaveraging Sizes")
@param_labels(["Model Name", "Start Size", "End Size", "Step Size", "Number of Epochs", "Learning Rate", "Stopping Criteria"])
def GLOBAL_train_with_multiple_subaveraging(
    model_name: str,
    start_size: int = 1,
    end_size: int = 10,
    step_size: int = 1,
    num_epochs: int = 20,
    lr: float = 1e-3,
    stopping_criteria: bool = False
):
    """
    Trains a model on multiple subaveraging sizes from start_size to end_size (inclusive) 
    by step_size increments. Each subaveraging configuration is trained independently.
    """
    import copy
    
    print(f"\n{'='*80}")
    print(f"Training with subaveraging sizes from {start_size} to {end_size} (step={step_size})")
    print(f"{'='*80}\n")
    
    results = {}
    
    # Iterate through each subaveraging size
    for size in range(start_size, end_size + 1, step_size):
        print(f"\n{'='*80}")
        print(f"Subaveraging Size: {size}")
        print(f"{'='*80}\n")
        
        # Create deep copies of subjects for this subaveraging size
        subaveraged_subjects = []
        for subject in SUBJECTS:
            # Create a deep copy to avoid modifying the original
            subject_copy = copy.deepcopy(subject)
            # Ensure device is set (copy the device attribute if it exists)
            if hasattr(subject, 'device'):
                subject_copy.device = subject.device
            else:
                # Default to CPU if device was never set
                subject_copy.setDevice(use_gpu=False)
            # Apply subaveraging
            subject_copy.subaverage(size=size)
            subaveraged_subjects.append(subject_copy)
        
        # Train on each subaveraged subject
        size_results = []
        for idx, subject in enumerate(subaveraged_subjects):
            print(f"\n--- Subject {idx + 1}/{len(subaveraged_subjects)} (Subavg Size={size}) ---")
            subject.train(
                model_name=model_name,
                num_epochs=num_epochs,
                lr=lr,
                output_dir=f"outputs/subavg_{size}_subject_{idx+1}",
                stopping_criteria=stopping_criteria
            )
        
        results[size] = {
            "num_subjects": len(subaveraged_subjects),
            "total_trials": sum(len(s.trials) for s in subaveraged_subjects)
        }
    
    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    for size, info in results.items():
        print(f"Subavg Size {size}: {info['num_subjects']} subjects, {info['total_trials']} total trials")
    print(f"{'='*80}\n")

