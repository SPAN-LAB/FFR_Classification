"""
SPAN Lab - FFR Classification

Filename: eeg_trial.py
Author(s): Kevin Chen
Description: The interface and implementation of the EEGTrial type.
    EEGTrial represents a collection of the data associated with a single trial. 
"""

from __future__ import annotations
from numpy import typing as npt
import numpy as np


class EEGTrial:
    
    # MARK: Initializer and stored properties
    
    def __init__(
        self,
        subject,
        data,
        trial_index,
        timestamps,
        raw_label,
        mapped_label=None,
        prediction=None,
        prediction_distribution=None,
    ):
        self.subject = subject
        self.data = data
        self.trial_index = trial_index
        self.timestamps = timestamps
        self.raw_label = raw_label
        self.mapped_label = mapped_label
        self.prediction = prediction
        self.prediction_distribution = prediction_distribution

        # Use default label-grabbing behavior
        self._label_preference = None

    # MARK: Computed properties

    @property
    def label(self):
        if self._label_preference is None:
            if self.mapped_label is None:
                return self.raw_label
            return self.mapped_label
        elif self._label_preference == "raw":
            return self.raw_label
        elif self._label_preference == "mapped":
            return self.mapped_label
        else:
            raise ValueError("Unrecognized label preference")
    
    @property
    def enumerated_label(self):
        return self.subject.labels_map[self.label]

    # MARK: Label management

    def set_label_preference(self, pref: str | None = None):
        """
        "raw", "mapped", or None
        """
        self._label_preference = pref
    
    def unemumerated(self, *, enumerated_label: int) -> any:
        unenumerating_label_map = self.subject.get_unenumerating_label_map()
        if enumerated_label in unenumerating_label_map:
            return unenumerating_label_map[enumerated_label]
        else:
            raise ValueError("Bad label!")
    
    def enumerated(self, *, unenumerated_label: int) -> int:
        enumerating_label_map = self.subject.get_enumerating_label_map()
        if unenumerated_label in enumerating_label_map:
            return enumerating_label_map[unenumerated_label]
        else:
            raise ValueError("Bad label!")            
    
    # MARK: Prediction management

    # WARNING: This method has been deprecated! Use set_prediction_distribution
    def set_prediction(self, enumerated_label):
        print("The set_prediction method has been deprecated! Use \"set_prediction_distribution\"")
        reversed_labels_map = {value: key for key, value in self.subject.labels_map.items()}
        self.prediction = reversed_labels_map[enumerated_label]
    
    def set_prediction_distribution(self, *, enumerated_prediction_distribution):
        
        best_unenumerated_label = None
        highest_probability = 0
        prediction_distribution = {}
        
        for enumerated_label, probability in enumerated_prediction_distribution.items():
            
            unenumerated_label = self.unemumerated(enumerated_label=enumerated_label)
            prediction_distribution[unenumerated_label] = probability
            
            if probability > highest_probability:
                best_unenumerated_label = unenumerated_label
                highest_probability = probability
        
        self.prediction_distribution = prediction_distribution
        self.prediction = best_unenumerated_label

    # MARK: Transformations
    
    def trim_by_index(self, start_index: int, end_index: int):
        self.data = self.data[start_index : end_index + 1]
        self.timestamps = self.timestamps[start_index : end_index + 1]

    def trim_by_timestamp(self, start_time: float, end_time: float):
        start = int(np.searchsorted(self.timestamps, start_time, side="left"))
        end = int(np.searchsorted(self.timestamps, end_time, side="right"))
        self.timestamps = self.timestamps[start:end]
        self.data = self.data[start:end]

    def __len__(self):
        return len(self.timestamps)
    
    @staticmethod
    def get_accuracy(trials: list[EEGTrial] | list[list[EEGTrial]]) -> float:
        
        if len(trials) == 0:
            print("Warning: No trials supplied.")
            return 0
        
        num_correct = 0
        total = 0
        
        # list[EEGTrial]
        if isinstance(trials[0], EEGTrial):
            total = len(trials)
            for trial in trials:
                if trial.prediction == trial.label:
                    num_correct += 1
        # list[list[EEGTrial]]
        else:
            for trial_list in trials:
                for trial in trial_list:
                    total += 1
                    if trial.prediction == trial.label:
                        num_correct += 1

        return num_correct / total
