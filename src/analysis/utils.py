"""
SPAN Lab - FFR Classification

Filename: utils.py
Author(s): Kevin Chen
Description: Utility functions used in the analysis code.
"""


import os
from pathlib import Path
import pickle

from ..core import AnalysisPipeline, PipelineState, EEGSubject
from ..core.ffr_proc import get_accuracy


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
            accuracy = get_accuracy(subject)
            
            axes.append((subaverage_size, accuracy))

    return axes