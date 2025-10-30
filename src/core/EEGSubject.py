from __future__ import annotations
from typing import Any, Self, Callable
from abc import ABC, abstractmethod

from .EEGTrial import EEGTrialInterface, EEGTrial

import numpy as np
from pymatreader import read_mat
from random import shuffle

class EEGSubjectInterface(ABC):
    trials: list[EEGTrialInterface]
    source_filepath: str
    folds: list[list[EEGTrialInterface]]

    @staticmethod
    @abstractmethod
    def init_from_filepath(filepath: str) -> Self: ... 

    @abstractmethod
    def trim_by_index(self, start_index: int, end_index: int) -> Self: ... 

    @abstractmethod
    def trim_by_timestamp(self, start_time: float, end_time: float) -> Self: ...

    @abstractmethod
    def subaverage(self, size: int) -> Self: ... 

    @abstractmethod
    def fold(self, num_folds: int) -> Self: ...
    
    @abstractmethod
    def map_trial_labels(self, rule_filepath: str) -> Self: ...

    @abstractmethod
    def grouped_trials(self, key: Callable[[EEGTrialInterface], Any]) -> dict[any, list[EEGTrialInterface]]: ...

class EEGSubject(EEGSubjectInterface):
    def __init__(
        self,
        *,
        trials=None,
        source_filepath=None
    ):
        """
        Provide argument for either `trials` or `source_filepath` but not both.
        """
        self.trials = trials
        self.source_filepath = source_filepath
        self.folds = None

    @staticmethod
    def init_from_filepath(filepath: str) -> Self:
        # Get the raw data
        raw = read_mat(filepath)
        raw_data = raw["ffr_nodss"]
        timestamps = raw["time"]
        labels = raw["#subsystem#"]["MCOS"][3]

        # Create the EEGTrial instances
        trials = []
        for i, trial in enumerate(raw_data):
            trials.append(EEGTrial(
                data=trial,
                timestamps=timestamps,
                trial_index=i,
                raw_label=labels[i]
            ))
        
        return EEGSubject(trials=trials, source_filepath=filepath)
    
    def trim_by_index(self, start_index: int, end_index: int) -> Self:
        for trial in self.trials:
            trial.trim_by_index(start_index, end_index)
        return self

    def trim_by_timestamp(self, start_time: float, end_time: float) -> Self:
        for trial in self.trials:
            trial.trim_by_timestamp(start_time, end_time)
        return self
    
    def subaverage(self, size: int) -> Self:
        grouped_trials = self.grouped_trials(key=lambda trial: trial.mapped_label)
        subaveraged_trials = [] 

        for _, trial_group in grouped_trials.items():
            shuffle(trial_group)
            n = len(trial_group)

            for i in range(0, n, size):
                chunk = trial_group[i:i + size]

                # Skip incomplete last group
                if len(chunk) < size:
                    continue

                stacked_data = np.array([trial.data for trial in chunk])
                subaveraged_data = np.mean(stacked_data, axis=0)

                subaveraged_trial = EEGTrial(
                    data=subaveraged_data,
                    trial_index=len(subaveraged_trials),
                    timestamps=chunk[0].timestamps,
                    raw_label=chunk[0].raw_label,
                    mapped_label=chunk[0].mapped_label
                )
                subaveraged_trials.append(subaveraged_trial)

        self.trials = subaveraged_trials
        return self

    def fold(self, num_folds: int) -> Self:
        folds = [[] for _ in range(num_folds)]
        grouped_trials = self.grouped_trials(key=lambda trial: trial.mapped_label)

        for _, trial_group in grouped_trials.items():
            shuffle(trial_group)
            for i, trial in enumerate(trial_group):
                folds[i % num_folds].append(trial)
        self.folds = folds
        return self

    def use_raw_labels(self):
        for tr in self.trials:
            tr.mapped_label = tr.raw_label
    
    def map_trial_labels(self, rule_filepath: str) -> Self:        
        # Create a dictionary that maps from raw label to mapped label
        labels_map: dict[int, int] = {}
    
        with open(rule_filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines or comments

                values = line.split(',')
                mapped_label = int(values[0].strip())  # First value is mapped label
                for raw_label in values[1:]:
                    raw_label = raw_label.strip()
                    if raw_label:
                        labels_map[int(raw_label)] = mapped_label  # Convert raw labels to int

        # Assign mapped labels to each trial
        for trial in self.trials:
            raw = int(trial.raw_label)  # Ensure raw_label is int
            if raw not in labels_map:
                raise ValueError(f"Raw label {raw} not found in mapping.")
            trial.mapped_label = labels_map[raw]

        return self
        
    def grouped_trials(
        self, 
        key: Callable[[EEGTrial], Any]=lambda trial: trial.mapped_label
    ) -> dict[any, list[EEGTrial]]:
        # Divide into groups separated by their label (accessed by the key)
        g = {}
        for trial in self.trials:
            if key(trial) in g: 
                g[key(trial)].append(trial)
            else:
                g[key(trial)] = [trial]
        return g