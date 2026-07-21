"""
SPAN Lab - FFR Classification

Filename: eeg_subject.py
Author(s): Kevin Chen
Description: The interface and implementation of the EEGSubject type.
    EEGSubject primarily represents a collection of all the trial data recorded for some individual.
"""


from __future__ import annotations
from typing import Any, Self, Callable

import numpy as np
from pymatreader import read_mat
from pathlib import Path

from .eeg_trial import EEGTrial
import os
import sys
from copy import deepcopy

# from .utils import silence_stderr

class EEGSubject:
    
    # MARK: Initializer and stored properties
    
    def __init__(self, *, trials=None, source_filepath=None):
        """
        Provide argument for either `trials` or `source_filepath` but not both.
        """
        if trials is None:
            trials = []
        self.trials: list[EEGTrial] = trials
        self.source_filepath = source_filepath
        self.folds: list[list[EEGTrial]] | None = None
        self.labels_map = {}
        self.setup_labels_map()

    # MARK: Computed properties

    @property
    def name(self) -> str:
        if self.source_filepath is None:
            return "<Undeterminable name>"

        filename = Path(self.source_filepath).stem # Get filename from filepath (without extension)
        return filename

    @property
    def trial_size(self) -> int:
        return len(self.trials[0])

    @property
    def num_categories(self) -> int:
        return len(self.grouped_trials().keys())

    # MARK: IO

    @staticmethod
    def init_from_filepath(filepath: str, extract: Callable = None, data_var: str = "ffr_nodss") -> EEGSubject:
        def default_extract(raw_mat_file: dict[str, Any]) -> dict[str, any]:
            """
            Default method of extracting the data from the raw .mat file.
            :returns: a dictionary with keys "data", "timestamps", and "labels".
            """
            output = {}
            import numpy as np
            data = raw_mat_file[data_var]
            if isinstance(data, dict):
                data = list(data.values())[0]
            import numpy as np
            data = np.array(data)
            if isinstance(data, dict):
                # Some .mat versions wrap arrays in a dict — extract the array
                data = list(data.values())[0]
            import numpy as np
            data = np.array(data)
            # pymatreader may return (timepoints, trials) or (trials, timepoints)
            # ensure shape is (trials, timepoints) by matching labels count
            n_labels = len(raw_mat_file["labels"])
            if data.shape[0] != n_labels:
                data = data.T
            if data.shape[0] != n_labels:
                raise ValueError(f"Data shape {data.shape} doesn't match labels count {n_labels}")
            output["data"] = data
            output["timestamps"] = raw_mat_file["time"]
            labels = raw_mat_file["labels"]
            # If labels came back as uint32 array (MATLAB object references),
            # re-read using scipy which handles this correctly
            import numpy as np
            if isinstance(labels, np.ndarray) and labels.dtype == np.uint32:
                import h5py
                with h5py.File(filepath, 'r') as f:
                    mcos = f['#subsystem#']['MCOS']
                    # MCOS[3] contains per-trial labels as uint8 (1,2,3,4)
                    trial_labels = f[mcos[0][3]][0]  # shape (3837,)
                    labels = [str(l) for l in trial_labels]
            elif not isinstance(labels, list):
                labels = list(labels)
            output["labels"] = labels
            return output

        # Get the raw data from the .mat file
        # raw = None
        # def do(): 
        raw = read_mat(filepath)
        # raw = read_mat(filepath)
        # silence_stderr(do)

        # Use the default extraction method if one isn't provided
        if extract is None:
            extract = default_extract

        extracted_data = extract(raw)
        raw_data = extracted_data["data"]
        timestamps = extracted_data["timestamps"]
        labels = extracted_data["labels"]

        # Create the EEGTrial instances
        subject = EEGSubject()
        trials = []
        for i, trial in enumerate(raw_data):
            trials.append(
                EEGTrial(
                    subject=subject,
                    data=trial,
                    timestamps=timestamps,
                    trial_index=i,
                    raw_label=labels[i],
                )
            )
        subject.trials = trials
        subject.source_filepath = filepath
        subject.setup_labels_map()

        return subject

    # MARK: Processing methods

    def trim_by_index(self, start_index: int, end_index: int) -> EEGSubject:
        for trial in self.trials:
            trial.trim_by_index(start_index, end_index)
        return self

    def trim_by_timestamp(self, start_time: float, end_time: float) -> EEGSubject:
        for trial in self.trials:
            trial.trim_by_timestamp(start_time, end_time)
        return self

    def subaverage(self, size: int) -> EEGSubject:
        grouped_trials = self.grouped_trials()
        subaveraged_trials = []

        for _, trial_group in grouped_trials.items():
            # shuffle(trial_group)
            n = len(trial_group)

            for i in range(0, n, size):
                chunk = trial_group[i : i + size]

                # Skip incomplete last group
                if len(chunk) < size:
                    continue

                stacked_data = np.array([trial.data for trial in chunk])
                subaveraged_data = np.mean(stacked_data, axis=0)

                subaveraged_trial = EEGTrial(
                    subject=self,
                    data=subaveraged_data,
                    trial_index=len(subaveraged_trials),
                    timestamps=chunk[0].timestamps,
                    raw_label=chunk[0].raw_label,
                    mapped_label=chunk[0].mapped_label,
                )
                subaveraged_trials.append(subaveraged_trial)

        self.trials = subaveraged_trials
        return self

    def fold(self, num_folds: int) -> EEGSubject:
        folds = [[] for _ in range(num_folds)]
        grouped_trials = self.grouped_trials()

        for _, trial_group in grouped_trials.items():
            # shuffle(trial_group)
            if len(trial_group) < num_folds:
                raise ValueError("""Fewer trials in one category than num_folds.
This causes some folds to have 0 trials from this category.""")
            for i, trial in enumerate(trial_group):
                folds[i % num_folds].append(trial)
        self.folds = folds
        return self

    def map_trial_labels(self, rule_filepath: str) -> Self:
        # Create a dictionary that maps from raw label to mapped label
        labels_map: dict = {}

        with open(rule_filepath, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines or comments
                values = line.split(",")
                raw_mapped = values[0].strip()  # Try int, then float, then str
                try:
                    mapped_label = int(raw_mapped)
                except ValueError:
                    try:
                        mapped_label = float(raw_mapped)
                    except ValueError:
                        mapped_label = raw_mapped
                for raw_label in values[1:]:
                    raw_label = raw_label.strip()
                    if raw_label:
                        try:
                            key = int(raw_label)
                        except ValueError:
                            try:
                                key = float(raw_label)
                            except ValueError:
                                key = raw_label
                        labels_map[key] = mapped_label

        # Assign mapped labels to each trial
        for trial in self.trials:
            raw = trial.raw_label  # Ensure raw_label is int
            if raw not in labels_map:
                try:
                    raw = int(raw)
                except (ValueError, TypeError):
                    pass
            if raw not in labels_map:
                try:
                    raw = float(raw)
                except (ValueError, TypeError):
                    pass
            if raw not in labels_map:
                raw = str(raw)
            if raw not in labels_map:
                raise ValueError(f"Raw label {trial.raw_label} not found in mapping.")
            trial.mapped_label = labels_map[raw]
        return self

    # MARK: Label management

    def set_label_preference(self, pref: str | None = None):
        for trial in self.trials:
            trial.set_label_preference(pref)

    def setup_labels_map(self):
        # Find all the labels
        self.labels_map = {}
        labels_set = set()
        labels_array = []

        for trial in self.trials:
            if trial.label not in labels_set:
                labels_array.append(trial.label)
                labels_set.add(trial.label)

        for i, label in enumerate(labels_array):
            self.labels_map[label] = i
    
    def get_unenumerating_label_map(self) -> dict[int, any]:
        unenumerating_label_map = {}
        for label, enumerated_label in self.labels_map.items():
            unenumerating_label_map[enumerated_label] = label
        return unenumerating_label_map
    
    def get_enumerating_label_map(self) -> dict[any, int]:
        return self.labels_map
            
    # MARK: Helpers

    def grouped_trials(self) -> dict[any, list[EEGTrial]]:
        # Divide into groups separated by their label
        g = {}
        for trial in self.trials:
            if trial.label in g:
                g[trial.label].append(trial)
            else:
                g[trial.label] = [trial]
        return g
    
    def reindex_trials(self):
        for i, trial in enumerate(self.trials):
            trial.trial_index = i
            
    @staticmethod
    def create_merged(*, subjects: list[EEGSubject]) -> EEGSubject:
        
        if len(subjects) == 0:
            return
        
        # Gather all trials
        all_trials = []
        for subject in subjects:
            for trial in subject.trials:
                all_trials.append(deepcopy(trial))
        
        merged_subject = EEGSubject(all_trials, source_filepath="DNE")
        return merged_subject
        