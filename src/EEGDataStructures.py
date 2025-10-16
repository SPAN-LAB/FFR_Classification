import numpy as np
from random import shuffle
import pandas as pd 

from numpy import typing as npt
from typing import *

# NOTE: This implementation was made for single-channel data. 
class EEGTrial:
    """
    Represents data from a single trial: data, timestamps, raw_label, mapped_label, and trial index.
    """
    def __init__(self, data: npt.NDArray, timestamps: npt.NDArray, raw_label: int, trial_index: int):
        """
        :param data: A 1D numpy array containing the data.
        :param timestamps: The clock time corresponding to each datapoint in `data`
        :param raw_label: This `EEGTrial`'s initial (unprocessed) label
        :param mapped_label: This `EEGTrial`'s actual (mapped) label 
        :param trial_index: This `EEGTrial`'s trial index
        """
        self.data = self.formatted_data(data)
        self.timestamps = self.formatted_timestamps(timestamps)
        self.raw_label = raw_label
        self.mapped_label = None # Initially None
        self.trial_index = trial_index

    def trim(self, start_index: int, end_index: int):
        """
        Trims `data` and `timestamps`, keeping only the values within the parameters' bounds. 
        """
        self.data = self.data[start_index: end_index + 1]

        # Trim the timestamps too for consistent lengths 
        self.timestamps = self.timestamps[start_index: end_index + 1]


    def map_labels(self, filename: str=None, map: Dict[int, int]=None): 
        """
        Maps from raw labels to class labels. Accepts either a filename or a dictionary for mapping.
        """
        if not map and not filename:
            # Used if the raw_label IS the intended mapped_label
            self.mapped_label = self.raw_label
            return
        elif not map:
            # Used if we want to use the filename to map
            map = self.create_map_from_csv(filename)
        
        # By this point, map is non-None
        if self.raw_label not in map:
            raise KeyError("The raw label was not found in the map.")
        self.mapped_label = map[self.raw_label]
    
    @staticmethod
    def create_map_from_csv(path: str) -> Dict[int, int]:
        """
        Creates a dictionary that maps from raw labels to actual labels. This function is used by 
        `map_labels`.

        :param path: the path to the CSV file
        """
        instructions = pd.read_csv(path)
        labels_map = {}
        for column_label in instructions.columns:
            for raw_label in instructions[column_label]:
                labels_map[raw_label] = int(column_label)
        return labels_map
                
    @staticmethod
    def formatted_data(data: npt.NDArray) -> npt.NDArray:
        """
        Raises an error if `data` is not 1D, and returns it otherwise.
        """
        if data.ndim != 1:
            raise ValueError("Only 1D arrays are permitted.")
        return data
    
    @staticmethod
    def formatted_timestamps(timestamps: npt.NDArray) -> npt.NDArray:
        """
        Raises an error if `data` is not 1D, and returns it otherwise.
        """
        if timestamps.ndim != 1:
            raise ValueError("Only 1D arrays are permitted.")
        return timestamps


class EEGSubject: 
    def __init__(self, trials: Sequence[EEGTrial]):
        self._trials = trials
        self._subaveraged_trials = None
    
    @property
    def trials(self):
        """
        Logic for using either `_trials` or `_subaveraged_trials`
        """
        if self._subaveraged_trials is not None:
            return self._subaveraged_trials
        return self._trials
    
    def subaverage(self, size: int):
        """
        Averages data values over `size` `EEGTrial`s who have the same `mapped_label`

        :param size: The number of trials each subaveraging is composed from. 

        TODO: Let the key be customizable (for example, use `raw_label` instead of `mapped_label`)
        """
        # Group trials by their labels 
        grouped_trials = self.grouped_trials()

        # Randomize the order of each list of homogeneous trials 
        for _, homogeneous_trials in grouped_trials.items():
            shuffle(homogeneous_trials)

        # Subaverage the trials 
        subaveraged_trials: List[EEGTrial] = []
        for label, homogeneous_trials in grouped_trials.items():
            n = len(homogeneous_trials)
            i = 0
            while i + size <= n:
                stacked_data = np.array([trial.data for trial in homogeneous_trials[i: i + size]])
                subaveraged_data = np.mean(stacked_data, axis=0)

                # Create a new EEGTrial object and assign it an artificial label
                # This artificial label equals the length of `subaveraged_trials`
                subaveraged_trial = EEGTrial(
                    data=subaveraged_data, 
                    timestamps=homogeneous_trials[0].timestamps, 
                    raw_label=label,
                    trial_index=len(subaveraged_trials)
                )
                subaveraged_trials.append(subaveraged_trial)

                i += size

        # Store the trials in `self.subaveraged_trials`
        self._subaveraged_trials = subaveraged_trials
            
    def test_split(self, trials: Sequence[EEGTrial], ratio: float):
        """
        Splits `trials` input into a train set and test set according to the specified `ratio`. 

        :param trials: the trials to be split into train and test sets
        :param ratio: the ratio of `trials` that become train sets 
        """
        if ratio > 1 or ratio < 0:
            raise ValueError("Ratio must be in [0, 1].")

        shuffle(trials) # Could be removed if we are guaranteed already-shuffled trials 
        cutoff_index = int(len(trials) * ratio)
        return trials[:cutoff_index], trials[cutoff_index:]

    def stratified_folds(self, num_folds):
        folds = [[] for i in range(num_folds)]
        grouped_trials = self.grouped_trials()

        # For each label, we shuffle then distribute them over the folds
        for _, homogeneous_trials in grouped_trials.items():
            shuffle(homogeneous_trials)
            for i, trial in enumerate(homogeneous_trials):
                folds[i % num_folds].append(trial)
        
        # Shuffle the trials in each fold 
        for fold in folds:
            shuffle(fold)
        
        return folds
    
    def grouped_trials(self) -> Dict[int, Sequence[EEGTrial]]:
        """
        A private helper method for grouping trials in a dictionary according to their labels. i.e.
        all trials with label = 1 are in grouped_trials[1]... 
        """
        grouped_trials: Dict[int, Sequence[EEGTrial]] = {}
        for trial in self.trials:
            if trial.mapped_label in grouped_trials:
                grouped_trials[trial.mapped_label].append(trial)
            else:
                grouped_trials[trial.mapped_label] = [trial]
        return grouped_trials