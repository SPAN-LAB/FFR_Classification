from __future__ import annotations
from typing import *
from utils import function_label, param_labels
from user_functions import SUBJECTS
from EEGDataStructures import EEGSubject

@function_label("Load Subject Data")
@param_labels(["Filepath"])
def GUI_load_subject_data(filepath: str):
    """
    Initializes a new EEGSubject instance using the given filepath.
    """
    # Check if the subject data has already been loaded.
    # If so, we don't do anything.
    for subject in SUBJECTS:
        if subject.source_filepath == filepath:
            return
    SUBJECTS.append(EEGSubject.init_from_filepath(filepath))

@function_label("Filter Trials")
@param_labels(["Subject Index", "Trials to remove"])
def GUI_filter_trials(subject_index: int, removed_trial_indices: str):
    # Convert removed_trial_indices from string of csvs to list of ints
    removed_trial_indices = [int(s) for s in removed_trial_indices.split(",")]

    # Sort removed trial indices from greatest to smallest 
    removed_trial_indices = sorted(removed_trial_indices, reverse=True)

    subject = SUBJECTS[subject_index - 1]
    for i in removed_trial_indices:
        subject.trials.pop(i)

