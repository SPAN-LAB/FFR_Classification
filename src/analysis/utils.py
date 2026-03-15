"""
SPAN Lab - FFR Classification

Filename: utils.py
Author(s): Kevin Chen
Description: Utility functions used in the analysis code.
"""


from copy import deepcopy
import os
from pathlib import Path
import pickle
from math import floor

from ..core import AnalysisPipeline, PipelineState, EEGSubject, EEGTrial

def get_mats(folder_path: str) -> list[str]:
    """
    Returns a list of the paths to the `.mat` files contained in specified folder.
    """
    if not os.path.isdir(folder_path):
        raise ValueError("The path provided is not a folder path.")

    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            files.append(os.path.join(folder_path, file))
    return files

def get_subject_loaded_pipelines(subject_filepaths: list[str]) -> dict[str, PipelineState]:
    """
    Returns a dictionary where each key is the filepath to a subject's data, and the value 
    associated with that key is an `AnalysisPipeline` object whose `subjects` attribute contains one
    `EEGSubject` which was loaded from the filepath. 
    
    Parameters
    ----------
    subject_filepaths : list[str]
        a list containing the filepaths to the subjects
    
    Returns
    -------
    dict[str, AnalysisPipeline]
        a dictionary where each value is an AnalysisPipeline loaded using its key
    """
    states = {}
    for subject_filepath in subject_filepaths:
        states[subject_filepath] = AnalysisPipeline().load_subjects(subject_filepath)
    return states

def save_times(indices, times, filepath):
    if len(indices) != len(times):
        raise ValueError("Arrays must have the same length.")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for x, y in zip(indices, times):
            f.write(f"{x},{y}\n")

def get_trailing_number(s: str) -> int | None:
    # Remove extension
    stem = Path(s).stem

    digits = []
    
    # Traverse from right to left
    for ch in reversed(stem):
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    if not digits:
        return None

    # digits were collected in reverse order
    return int("".join(reversed(digits)))

def get_results(dir_path: str) -> list[tuple[int, float]]:

    # For each pkl file in `dir_path`
    # 
    # - Determine the subaverage size and accuracy
    # - Place these results into the axes list
    # 
    # axes is a list of tuples (int, float) in the form 
    # (subaverage size, accuracy)
    
    # Get the names of all `.pkl` files in the provided directory
    dir = Path(dir_path)
    pkl_filenames = []
    for content in dir.glob("*.pkl"):
        if content.is_file():
            pkl_filenames.append(str(content))
    for content in dir.glob("*.json"):
        if content.is_file():
            pkl_filenames.append(str(content))
    print(f"Detected {len(pkl_filenames)} files.")

    # Determine the subaverage size and accuracy for each file
    axes = []
    # i = 0
    pkl_filenames.sort()
    for filename in pkl_filenames:
        with open(filename, "rb") as file:
            
            subject: EEGSubject = pickle.load(file)

            subaverage_size = get_trailing_number(filename)
            accuracy = EEGTrial.get_accuracy(subject)
            
            axes.append((subaverage_size, accuracy))
    
    axes.sort(key=lambda x: x[0])
    return axes

def stratified_deterministic_sample(subject: EEGSubject, num_trials: int) -> list[EEGTrial]:
    """
    Performs a stratified sample on the trials of the EEGSubject deterministically. 
    In other words, identical inputs to this function will provide identical outputs, and there 
    is no randomization involved. 
    """
    
    subject = deepcopy(subject)
    total_num_trials = len(subject.trials)
    grouped_trials = subject.grouped_trials()
    
    # Determine the number of trials each class needs 
    
    # Keys are any (all possible labels/classes of the subject).
    # Values are 2-element tuples, each of which represents the number of trials
    # this class should have, given the number of trials passed to this func.
    # The second element is the floored result of the first.
    num_trials_per_label = {}
    for label, trials in grouped_trials.items():
        n = len(trials) / total_num_trials * num_trials
        num_trials_per_label[label] = [n, floor(n)]
        
    # Determine the distance between the unfloored and floored values
    num_trials_allocated = 0
    distances = []
    for label, num_trials_tuple in num_trials_per_label.items():
        distances.append([label, num_trials_tuple[0] - num_trials_tuple[1]])
        num_trials_allocated += num_trials_tuple[1]
    distances.sort(key=lambda x: x[1], reverse=True)
    
    # Now, distances contains tuples of the form (Label, fractional_trials_needed)
    # sorted by the second value. So we can allocate remaining trials to 
    # classes whose priority is highest
    
    for distance in distances:
        if num_trials_allocated >= num_trials:
            break
        else:
            num_trials_per_label[distance[0]][1] += 1
            num_trials_allocated += 1
    
    if num_trials_allocated != num_trials:
        raise ValueError("Failed sample exactly the specified number of trials")
    
    # Now, the second element in the tuples (values) of num_trials_per_label 
    # corresponds to the number of trials we're going to use for that class
    
    sampled_trials = []
    for label, trials in grouped_trials.items():
        num_trials_to_take = num_trials_per_label[label][1]
        sampled_trials += grouped_trials[label][:num_trials_to_take]
    
    # Reassign indices to the trials 
    for i, trial in enumerate(sampled_trials):
        trial.trial_index = i
    
    return sampled_trials

def strip_data_away(subject: EEGSubject):
    for trial in subject.trials:
        trial.data = []
        trial.timestamps = []
    subject.folds = []
