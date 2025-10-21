from __future__ import annotations
from numpy import typing as npt
from typing import *
from enum import Enum, auto

import numpy as np
from random import shuffle
import pandas as pd 
from pymatreader import read_mat

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

    @property
    def label(self):
        if self.mapped_label is not None:
            return self.mapped_label
        else:
            return self.raw_label

    def trim(self, start_index: int, end_index: int):
        """
        Trims `data` and `timestamps`, keeping only the values included in the parameters' bounds. 
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
    class DataState(Enum):
        """
        This enum tracks the modification state of this instance. This also makes it easier to 
        determine whether when a method of this instance can or cannot run, because one just checks 
        the state to get information about the condition of the other attributes. (For example, 
        one cannot train with cross validation without first having split the data into folds.)
        """
        
        # The data has not been modified
        UNMODIFIED = auto() 
        
        # The data has been subaveraged. 
        # This comes with the guarantee that`self._subaveraged_trials` is not None.
        # Mutually exclusive with `UNMODIFIED`.
        SUBAVERAGED = auto() 

        # The data has been split into folds.
        # This comes with the guarantee that `self._folds` is not None.
        FOLDED = auto() 

    def is_unmodified(self):
        return self.DataState.UNMODIFIED in self.state
    
    def is_subaveraged(self):
        return self.DataState.SUBAVERAGED in self.state
    
    def is_folded(self):
        return self.DataState.FOLDED in self.state
    
    def remove_state(self, state: DataState):
        self.state.remove(state)

    def add_state(self, state: DataState):
        # Don't allow adding `UNMODIFIED` state.
        if state is self.DataState.UNMODIFIED:
            raise ValueError("Cannot add `UNMODIFIED` state to the subject.")
        
        if state is self.DataState.SUBAVERAGED:
            self.state.add(self.DataState.SUBAVERAGED)
            if self.is_unmodified():
                self.remove_state(self.DataState.UNMODIFIED)
        elif state is self.DataState.FOLDED:
            if self.is_unmodified():
                self.remove_state(self.DataState.UNMODIFIED)

    def __init__(self, trials: Sequence[EEGTrial]):
        self.state = set([self.DataState.UNMODIFIED])
        self._trials = trials
        self._subaveraged_trials = None
        self.folds = None

    @property
    def trials(self):
        """
        Logic for using either `_trials` or `_subaveraged_trials`
        """
        if self.DataState.SUBAVERAGED in self.state:
            # The data has been subaveraged
            return self._subaveraged_trials
        else: 
            # The data has not been subaveraged
            return self._trials

    @staticmethod
    def init_from_filepath(path: str) -> EEGSubject:
        # Open the file 
        file = read_mat(path)

        # Get the raw data
        raw_data = file["ffr_nodss"]
        timestamps = file["time"]
        labels = file["#subsystem#"]["MCOS"][3]

        # Create the EEGTrial instances
        trials = []
        for i, trial in enumerate(raw_data):
            trials.append(EEGTrial(
                data=raw_data[i],
                timestamps=timestamps,
                raw_label=labels[i],
                trial_index=i
            ))
        return EEGSubject(trials)
    
    def map_labels(self, filepath: str):
        labels_map = EEGTrial.create_map_from_csv(filepath)
        for trial in self.trials:
            trial.map_labels(map=labels_map)

    def trim(self, start_index, end_index):
        for trial in self.trials:
            trial.trim(start_index=start_index, end_index=end_index)

    def subaverage(self, size: int):
        """
        Averages data values over `size` `EEGTrial`s who have the same `mapped_label`

        :param size: The number of trials each subaveraging is composed from. 

        TODO: Let the key be customizable (for example, use `raw_label` instead of `mapped_label`)
        """
        print(f"Length before subaveraging: {len(self.trials)}")
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

                # Create a new EEGTrial object and assign it an artificial trial index
                # which equals the length of `subaveraged_trials`
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

        # Update the state
        self.add_state(self.DataState.SUBAVERAGED)

        print(f"Length after subaveraging: {len(self.trials)}")
            
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
        
        # Update the state
        self.add_state(self.DataState.FOLDED)

        # Store the folds
        self.folds = folds

        print(f"Folded into {len(folds)} folds")
    
    def grouped_trials(self) -> Dict[int, Sequence[EEGTrial]]:
        """
        A private helper method for grouping trials in a dictionary according to their labels. i.e.
        all trials with label = 1 are in grouped_trials[1]... 
        """
        grouped_trials: Dict[int, Sequence[EEGTrial]] = {}
        for trial in self.trials:
            if trial.label in grouped_trials:
                grouped_trials[trial.label].append(trial)
            else:
                grouped_trials[trial.label] = [trial]
        return grouped_trials

if __name__ == "__main__":
    e: EEGSubject = EEGSubject.init_from_filepath("/Users/kevin/Desktop/Work/SPAN_Lab/trial-classification/data/4T1002.mat")
    print(len(e._trials))

    s = e.subaverage

    s(size=5)

    print(len(e._subaveraged_trials))

