"""
SPAN Lab - FFR Classification

Filename: eeg_subject.py
Author(s): Kevin Chen
Description: The interface and implementation of the EEGSubject type.
    EEGSubject primarily represents a collection of all the trial data recorded for some individual.
"""


from __future__ import annotations
from typing import Any, Self, Callable
import ast

import numpy as np
from pymatreader import read_mat
from pathlib import Path

from .eeg_trial import EEGTrial
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
    def init_from_filepath(
        filepath: str,
        extract: Callable = None,
        data_var: str = "ffr_nodss",
    ) -> EEGSubject:
        def default_extract(raw_mat_file: dict[str, Any]) -> dict[str, any]:
            """
            Default method of extracting the data from the raw .mat file.

            :returns: a dictionary with keys "data", "timestamps", and "labels".
            """
            output = {}
            if data_var not in raw_mat_file:
                available = ", ".join(
                    key
                    for key in raw_mat_file.keys()
                    if not str(key).startswith("__")
                )
                raise ValueError(
                    f"Data variable '{data_var}' not found in {filepath}. "
                    f"Available variables: {available}"
                )

            data = raw_mat_file[data_var]
            while isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            data = np.asarray(data)

            labels = raw_mat_file["labels"]
            if isinstance(labels, np.ndarray) and labels.dtype == np.uint32:
                import h5py

                with h5py.File(filepath, "r") as file:
                    mcos = file["#subsystem#"]["MCOS"]
                    trial_labels = file[mcos[0][3]][0]
                    labels = [str(label) for label in trial_labels]
            elif isinstance(labels, np.ndarray):
                labels = labels.tolist()

            n_labels = len(labels)
            if data.ndim < 2:
                raise ValueError(
                    f"Data variable '{data_var}' must be at least 2D; got shape {data.shape}."
                )
            if data.shape[0] != n_labels and data.shape[-1] == n_labels:
                data = data.T
            if data.shape[0] != n_labels:
                raise ValueError(
                    f"Data variable '{data_var}' shape {data.shape} does not match "
                    f"labels count {n_labels}."
                )

            output["data"] = data
            output["timestamps"] = raw_mat_file["time"]
            output["labels"] = labels
            return output

        raw = read_mat(filepath)

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

    @staticmethod
    def _normalize_label(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.shape == () or value.size == 1:
                return value.item()
            return tuple(value.tolist())
        if isinstance(value, list):
            return tuple(value)
        return value

    @staticmethod
    def parse_label_token(value: Any) -> Any:
        if not isinstance(value, str):
            return EEGSubject._normalize_label(value)

        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        return EEGSubject._normalize_label(parsed)

    @staticmethod
    def _label_key(value: Any) -> Any:
        return EEGSubject._normalize_label(value)

    def trim_by_index(self, start_index: int, end_index: int) -> EEGSubject:
        for trial in self.trials:
            trial.trim_by_index(start_index, end_index)
        return self

    def trim_by_timestamp(self, start_time: float, end_time: float) -> EEGSubject:
        for trial in self.trials:
            trial.trim_by_timestamp(start_time, end_time)
        return self

    def trim_by_type(
        self,
        label_values: str | list[Any],
        label_source: str = "raw",
    ) -> EEGSubject:
        if isinstance(label_values, str):
            label_values = [
                value.strip()
                for value in label_values.replace(";", ",").split(",")
                if value.strip()
            ]

        allowed = {
            self._label_key(self.parse_label_token(value))
            for value in label_values
        }

        def label_for(trial: EEGTrial):
            if label_source == "raw":
                return trial.raw_label
            if label_source == "mapped":
                return trial.mapped_label
            if label_source == "current":
                return trial.label
            raise ValueError("label_source must be 'raw', 'mapped', or 'current'.")

        self.trials = [
            trial
            for trial in self.trials
            if self._label_key(label_for(trial)) in allowed
        ]
        self.folds = None
        self.setup_labels_map()
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
        labels_map: dict[Any, Any] = {}

        with open(rule_filepath, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines or comments

                values = line.split(",")
                mapped_label = self.parse_label_token(values[0])
                for raw_label in values[1:]:
                    raw_label = raw_label.strip()
                    if raw_label:
                        labels_map[self._label_key(self.parse_label_token(raw_label))] = mapped_label

        # Assign mapped labels to each trial
        for trial in self.trials:
            raw = self._label_key(trial.raw_label)
            if raw not in labels_map:
                raise ValueError(f"Raw label {raw} not found in mapping.")
            trial.mapped_label = labels_map[raw]

        self.setup_labels_map()
        return self

    # MARK: Label management

    def set_label_preference(self, pref: str | None = None):
        for trial in self.trials:
            trial.set_label_preference(pref)

    def setup_labels_map(self):
        self.labels_map = {}
        # Find all the labels
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

    def grouped_trials(
        self,
        key: Callable[[EEGTrial], Any] | None = None,
    ) -> dict[any, list[EEGTrial]]:
        # Divide into groups separated by their label
        if key is None:
            key = lambda trial: trial.label
        g = {}
        for trial in self.trials:
            group_key = key(trial)
            if group_key in g:
                g[group_key].append(trial)
            else:
                g[group_key] = [trial]
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
        
        merged_subject = EEGSubject(trials=all_trials, source_filepath="DNE")
        return merged_subject
        
