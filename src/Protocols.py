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
        pass

    def is_subaveraged(self):
        pass

    def is_folded(self):
        pass

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

    @property
    def stratified_folds(self, num_folds: int) -> list[list[EEGTrialProtocol]]:
        """
        IMPORTANT NOTE: Call as `self.stratified_folds`, not `self.stratified_folds()`.

        Splits `self.trials` into `num_folds` folds in a stratified manner. Each
        sub-list of trials in the output has approximately equivalent size.
        """
        ...

    def __init__(self, filepath: str):
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

class EEGTrial(EEGTrialProtocol):
    def __init__(
        self, 
        data: npt.NDArray, 
        timestamps: npt.NDArray, 
        trial_index: int,
        raw_label: any,
        mapped_label: any=None,
    ):
        self.data = data
        self.timestamps = timestamps
        self.trial_index = trial_index
        self.raw_label = raw_label
        self.mapped_label = mapped_label
    
class EEGSubject(EEGSubjectProtocol):
    def __init__(self, filepath: str):
        self.state = EEGSubjectStateTracker()
        self.trials: list[EEGTrialProtocol] = []
        self.source_filepath = filepath

        # Internal attribute for storing folds
        self._folds: list[list[EEGTrialProtocol]] | None = []

        self.load_data(filepath)
    
    def load_data(self, filepath: str):
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
    
    @property
    def stratified_folds(self, num_folds: int=1) -> list[list[EEGTrialProtocol]]:
        # Skip recalculation if the data has already been split into folds
        if (self._folds is not None) and (len(self._folds) == num_folds):
            return self._folds
        
        folds = [[] for _ in range(num_folds)]
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
        
        self.folds = folds
        return folds
    
    def subaverage(self, size: int): 
        grouped_trials = self.grouped_trials()
        
        subaveraged_trials: list[EEGTrial] = []
        for label, trial_group in grouped_trials.items():
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
            if trial.label in grouped_trials:
                grouped_trials[trial.label].append(trial)
            else:
                grouped_trials[trial.label] = [trial]
        return grouped_trials
            
