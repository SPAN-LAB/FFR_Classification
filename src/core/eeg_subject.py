from __future__ import annotations
from typing import Any, Self, Callable

from .eeg_trial import EEGTrialInterface, EEGTrial

import numpy as np
from pymatreader import read_mat
from random import shuffle

class EEGSubjectInterface:
    trials: list[EEGTrialInterface]
    source_filepath: str
    folds: list[list[EEGTrialInterface]]

    @property
    def trial_size(self):
        raise NotImplementedError("Implement this method.")

    @property
    def num_categories(self):
        raise NotImplementedError("Implement this method.")

    def set_label_preference(self, pref: str | None):
        raise NotImplementedError("Implement this method.")

    @staticmethod
    def init_from_filepath(filepath: str, extract: Callable | None) -> EEGSubject: 
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")

    def trim_by_index(self, start_index: int, end_index: int) -> EEGSubject: 
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")

    def trim_by_timestamp(self, start_time: float, end_time: float) -> EEGSubject: 
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")

    def subaverage(self, size: int) -> EEGSubject:
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")

    def fold(self, num_folds: int) -> EEGSubject:
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")
    
    def map_trial_labels(self, rule_filepath: str) -> EEGSubject:
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")

    def grouped_trials(self) -> dict[any, list[EEGTrialInterface]]:
        """
        TODO
        """
        raise NotImplementedError("Implement this method.")

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
    def init_from_filepath(filepath: str, extract: Callable = None) -> EEGSubject:

        def default_extract(raw_mat_file: dict[str, Any]) -> dict[str, any]:
            """
            Default method of extracting the data from the raw .mat file. 

            :returns: a dictionary with keys "data", "timestamps", and "labels".
            """
            output = {}
            output["data"] = raw_mat_file["ffr_nodss"].T
            output["timestamps"] = raw_mat_file["time"]
            output["labels"] = raw_mat_file["labels"]
            return output

        # Get the raw data from the .mat file
        raw = read_mat(filepath)
        
        # Use the default extraction method if one isn't provided
        if extract is None:
            extract = default_extract
        
        extracted_data = extract(raw)
        raw_data = extracted_data["data"]
        timestamps = extracted_data["timestamps"]
        labels = extracted_data["labels"]

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
    
    def trim_by_index(self, start_index: int, end_index: int) -> EEGSubject:
        for trial in self.trials:
            trial.trim_by_index(start_index, end_index)
        return self

    def trim_by_timestamp(self, start_time: float, end_time: float) -> EEGSubject:
        for trial in self.trials:
            trial.trim_by_timestamp(start_time, end_time)
        return self
    
    def subaverage(self, size: int) -> EEGSubject:
        grouped_trials = self.grouped_trials(key=lambda trial: trial.raw_label)
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

    def fold(self, num_folds: int) -> EEGSubject:
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
        
    def grouped_trials(self) -> dict[any, list[EEGTrial]]:
        # Divide into groups separated by their label 
        g = {}
        for trial in self.trials:
            if trial.label in g: 
                g[trial.label].append(trial)
            else:
                g[trial.label] = [trial]
        return g
    
    @property
    def trial_size(self):
        return len(self.trials[0])
    
    def set_label_preference(self, pref: str | None = None):
        for trial in self.trials:
            trial.set_label_preference(pref)
    
    @property
    def num_categories(self):
        return len(self.grouped_trials().keys())