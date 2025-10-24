from __future__ import annotations
from typing import Protocol, Any
import numpy.typing as npt
from enum import Enum, auto


from pymatreader import read_mat
from random import shuffle
import numpy as np

class EEGSubjectState(Enum):
    """
    TODO
    """

    # The data has not been modified
    UNMODIFIED = auto()

    # The data has been subaveraged
    SUBAVERAGED = auto()

    # The data has been split into folds
    FOLDED = auto()

class EEGSubjectStateTracker:
    def __init__(self):
        self.state_set = set([EEGSubjectState.UNMODIFIED])
    
    def mark_subaveraged(self):
        self.state_set.add(EEGSubjectState.SUBAVERAGED)
        if EEGSubjectState.UNMODIFIED in self.state_set:
            self.state_set.remove(EEGSubjectState.UNMODIFIED)
    
    def mark_folded(self):
        self.state_set.add(EEGSubjectState.FOLDED)
        if EEGSubjectState.UNMODIFIED in self.state_set:
            self.state_set.remove(EEGSubjectState.UNMODIFIED)

    def is_modified(self):
        return EEGSubjectState.UNMODIFIED not in self.state_set

    def is_subaveraged(self):
        return EEGSubjectState.SUBAVERAGED in self.state_set

    def is_folded(self):
        return EEGSubjectState.FOLDED in self.state_set

Label = Any

class EEGTrialProtocol(Protocol):
    """
    TODO
    """

    # A 1D array containing the data
    data: npt.NDArray

    # A 1D array containing the timestamps. 
    # The length of this array should equal that of the data, since each data point is
    # associated with a timestamp. 
    timestamps: npt.NDArray

    # This trial's index
    trial_index: int

    # The raw label for this trial
    raw_label: Label

    _mapped_label: Label | None

    @property
    def mapped_label(self):
        """
        Returns `_mapped_label` if it exists; returns `raw_label` otherwise. 
        """

    def map_label(self, label: Label): 
        """
        Sets `_mapped_label` for this `EEGTrial`.
        """
    
    def trim(self, start_index: int, end_index: int):
        """
        Keeps only the `start_index`th to `end_index`th datapoints of `data` and `timestamps` 
        (inclusive).

        Thus requires that 0 <= `start_index` <= `end_index` < `len(data)`
        """
        ...
    
    def trim_interval(self, left_bound: float, right_bound: float):
        """
        Keeps the timestamps inside [left_bound, right_bound]. 
        """
        ...


class EEGSubjectProtocol(Protocol):
    """
    Represent the EEG data collected from one subject.
    """

    # A tracker for this instance's current state. The options are: 
    # - UNMODIFIED
    # - SUBAVERAGED
    # - FOLDED
    state: EEGSubjectStateTracker

    # The trial data
    trials: list[EEGTrialProtocol]

    # The path to the file used to instantiate this EEGSubject.
    source_filepath: str

    def __init__(self, filepath: str):
        """
        Initialize from a filepath.
        """

    def stratified_folds(self, num_folds: int) -> list[list[EEGTrialProtocol]]:
        """
        Splits `self.trials` into `num_folds` folds in a stratified manner. Each
        sub-list of trials in the output has approximately equivalent size.
        """
        ...

    def map_labels(self, rule_filepath: str) -> EEGSubjectProtocol:
        """
        Maps labels using the path to a CSV file.

        Mutates this instance and returns itself.
        """
        ...

    def subaverage(self, size: int) -> EEGSubjectProtocol:
        """
        Pools `size` trials with the same labels and averages their data to create 
        an `EEGSubject` with the subaveraged data. Mutates `self.trails` and `self.state`. 
        If there are insufficient trials for a subaveraging 
        (e.g. 9 trials for a subaveraging size of 5), all the remaining trials are discarded.
        
        Mutates this instance and returns itself.
        """
        ...

    def trim(self, start_index: int, end_index: int) -> EEGSubjectProtocol:
        """
        Trims the data for all `EEGTrial` instances in `self.trials`.
        The start and end indices are inclusive.

        Mutates this instance and returns itself.
        """
        ...
    
    def test_split(self, trials: EEGTrialProtocol) -> EEGSubjectProtocol:
        """
        Splits `trials` into a training and test set.
        `ratio` is the ratio of the size of the training set to the size of `trials`
        """
        ...