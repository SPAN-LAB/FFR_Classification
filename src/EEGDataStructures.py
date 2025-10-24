from __future__ import annotations
from numpy import typing as npt
from typing import *
from enum import Enum, auto

import numpy as np
from random import shuffle
import pandas as pd 
from pymatreader import read_mat
import matplotlib.pyplot as plt
import seaborn as sns

from protocols import *


class EEGTrial(EEGTrialProtocol):
    def __init__(
        self, 
        data: npt.NDArray[Any], 
        timestamps: npt.NDArray[Any], 
        trial_index: int,
        raw_label: Label,
        mapped_label: Label=None,
    ):
        self.data = data
        self.timestamps = timestamps
        self.trial_index = trial_index
        self.raw_label = raw_label
        self._mapped_label = mapped_label
    
    def trim(self, start_index: int, end_index: int):
        self.data = self.data[start_index: end_index + 1]
        self.timestamps = self.timestamps[start_index: end_index + 1]
    

class EEGSubject(EEGSubjectProtocol):
    def __init__(self, filepath: str):
        self.state = EEGSubjectStateTracker()
        self.trials: list[EEGTrialProtocol] = []
        self.source_filepath = filepath

        # Internal attribute for storing folds
        self._folds: list[list[EEGTrialProtocol]] | None = []

        self.load_data(filepath)
    
    def load_data(self, filepath: str):
        """
        A private helper method for getting the data using a given filepath.
        """
        # Get the raw data
        raw = read_mat(filepath)
        raw_data = raw["ffr_nodss"]
        timestamps = raw["time"]
        labels = raw["#subsystem#"]["MCOS"][3]

        # Create the EEGTrial instances
        self.trials = []
        for i, trial in enumerate(raw_data):
            self.trials.append(EEGTrial(
                data=trial,
                timestamps=timestamps,
                trial_index=i,
                raw_label=labels[i]
            ))
    
    def stratified_folds(self, num_folds: int=1) -> list[list[EEGTrialProtocol]]:
        # Skip recalculation if the data has already been split into folds
        if (self._folds is not None) and (len(self._folds) == num_folds):
            return self._folds
        
        folds: list[list[EEGTrialProtocol]] = [[] for _ in range(num_folds)]
        grouped_trials = self.grouped_trials()

        for _, trial_group in grouped_trials.items():
            shuffle(trial_group)
            for i, trial in enumerate(trial_group):
                folds[i % num_folds].append(trial)
            
            # We shuffle the folds to ensure each fold has approximately the same size. 
            # Notice that if we didn't shuffle, 23 trials divided into 5 folds would normally leave
            # group sizes of [5, 5, 5, 5, 3]. The aim with shuffling the folds 
            # is to flatten that distribution: [5, 4, 5, 5, 4]
            shuffle(folds)

        for fold in folds:
            shuffle(fold)
        
        self.state.mark_folded()

        self._folds = folds
        return folds

    def map_labels(self, rule_filepath: str):
        # Create a dictionary that maps from raw label to mapped label
        instructions = pd.read_csv(rule_filepath)
        labels_map = {}
        for column_label in instructions.columns:
            for raw_label in instructions[column_label]:
                labels_map[raw_label] = int(column_label)
        
        # Set the mapped_label for each trial
        for trial in self.trials:
            trial.map_label(labels_map[trial.raw_label])
    
    def subaverage(self, size: int) -> EEGSubject: 
        grouped_trials = self.grouped_trials()
        
        subaveraged_trials: list[EEGTrialProtocol] = []
        for _, trial_group in grouped_trials.items():
            shuffle(trial_group)
            n = len(trial_group)
            for i in range(0, n // size, size):
                stacked_data = np.array([trial.data for trial in trial_group[i*size: (i+1)*size]])
                subaveraged_data = np.mean(stacked_data, axis=0)

                subaveraged_trials.append(EEGTrial(
                    data=subaveraged_data,
                    timestamps=trial_group[0].timestamps,
                    trial_index=len(subaveraged_trials),
                    raw_label=trial_group[0].mapped_label,
                    mapped_label=trial_group[0].mapped_label
                ))

        self.state.mark_subaveraged()

        self.trials = subaveraged_trials
        return self

    def grouped_trials(self) -> dict[Label, list[EEGTrialProtocol]]:
        """
        A private helper method for grouping trials in a dictionary according to their labels. i.e.
        all trials with label = 1 are in grouped_trials[1]... 
        """
        grouped_trials: dict[Label, list[EEGTrialProtocol]] = {}
        for trial in self.trials:
            if trial.mapped_label in grouped_trials:
                grouped_trials[trial.mapped_label].append(trial)
            else:
                grouped_trials[trial.mapped_label] = [trial]
        return grouped_trials
    
    def trim(self, start_index: int, end_index: int):
        for trial in self.trials:
            trial.trim(start_index, end_index)

    def test_split(self, trials: list[EEGTrialProtocol], ratio: float) -> tuple[list[EEGTrialProtocol], list[EEGTrialProtocol]]:
        if ratio > 1 or ratio < 0:
            raise ValueError("Ratio must be in [0, 1].")

        shuffle(trials) # Could be removed if we are guaranteed already-shuffled trials 
        cutoff_index = int(len(trials) * ratio)
        return trials[:cutoff_index], trials[cutoff_index:]


