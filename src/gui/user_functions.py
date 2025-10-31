from __future__ import annotations
from typing import *
from .utils import function_label, param_labels
from ..core.EEGSubject import EEGSubject
from ..core.EEGTrial import EEGTrial
import copy
from ..core.Trainer import Trainer
from ..core.Plots import Plots

ORIGINAL_SUBJECTS: List[EEGSubject] = []
SUBJECTS: List[EEGSubject] = []
TRAINERS: List[Trainer] = []

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

@function_label("Test Model")
@param_labels([])
def GLOBAL_test_model():
    for trainer in TRAINERS:
        trainer.test()

@function_label("Train Model") 
@param_labels(["Use GPU?", "Model Name", 
               "Number of Epochs", "Learning Rate",
                 "Stopping Criteria"])
def GLOBAL_train_model(use_gpu: bool,
                       model_name: str,
                       num_epochs: int = 20,
                       lr: float = 1e-3,
                       stopping_criteria: bool = True
                       ): 
    for subject in SUBJECTS:
        trainer = Trainer(subject = subject, model_name = model_name)
        trainer.train(use_gpu = use_gpu, num_epochs = num_epochs,
                            lr = lr, stopping_criteria = stopping_criteria)
        TRAINERS.append(trainer)

@function_label("Load Grand Subject")
@param_labels([])
def GLOBAL_grand_load_subject():
    """
    Creates a pseudo-subject by concatenating trials from all currently loaded
    subjects (in ORIGINAL_SUBJECTS) and replaces all subjects with this one.
    """
    if not ORIGINAL_SUBJECTS:
        print("Grand load: no subjects loaded.")
        return

    combined_trials: List[EEGTrial] = []
    next_index = 0
    for subj in ORIGINAL_SUBJECTS:
        for tr in getattr(subj, 'trials', []) or []:
            combined_trials.append(
                EEGTrial(
                    data=tr.data,
                    trial_index=next_index,
                    timestamps=tr.timestamps,
                    raw_label=tr.raw_label,
                    mapped_label=getattr(tr, 'mapped_label', None)
                )
            )
            next_index += 1

    if not combined_trials:
        print("Grand load: subjects contain no trials.")
        return

    pseudo = EEGSubject(trials=combined_trials, source_filepath="grand://combined")

    # Replace all subjects with the grand subject and reflect in both lists
    ORIGINAL_SUBJECTS.clear()
    SUBJECTS.clear()
    ORIGINAL_SUBJECTS.append(pseudo)
    SUBJECTS.append(copy.deepcopy(pseudo))
    print(f"Grand subject loaded with {len(pseudo.trials)} trials (replaced all subjects).")

######################

@function_label("Plot Subject Data")
@param_labels([])
def GLOBAL_plot_subject_data():
    for subject in SUBJECTS:
        Plots.plot_averaged_trials(subject)

@function_label("Plot Grand Average")
@param_labels(["Show components"])
def GLOBAL_plot_grand_average(show_components: bool=True):
    Plots.plot_grand_average(SUBJECTS, show_components=show_components)

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

