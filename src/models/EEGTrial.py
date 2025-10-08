import numpy as np
from numpy import typing as npt

from pymatreader import read_mat

from typing import *

class EEGTrial:
    """
    Represents data from a single trial. Contains the sample data, label, and sample index.
    """
    def __init__(self, data: npt.NDArray, times: npt.NDArray, label: int, trial_index: int):
        # This is a 2D matrix where rows represent channels and columns represent datapoints.
        # That is, the r-th row is a 1D array of datapoints collected from the r-th channel.
        self.data: npt.NDArray = data
        if self.data.ndim == 1:
            # Ensure a 2D shape
            self.data = self.data.reshape(1, -1)

        # This is a vector where the i-th element is the time at which the i-th datapoint in 
        # self.data was collected.
        self.times: npt.NDArray = times

        # The label / class for this trial 
        self.label: int = label

        # The index of this trial 
        self.trial_index: int = trial_index

class AveragedEEGTrial:
    """
    Represents the averaged data of multiple EEGTrial instances.
    """
    def __init__(self, trials: Sequence[EEGTrial], trial_index: int):
        stack = np.stack([trial.data for trial in trials])
        averaged_data = stack.mean(axis=0)
        self.data = averaged_data
        self.times = trials[0].times
        self.label = trials[0].label
        self.trial_index = trial_index

class EEGData: 
    """
    Represents all EEGTrial instances obtained from a subject. 
    """
    # def __init__(self, identifier: str, trials: Sequence[EEGTrial]):
    #     """
    #     Initialize from a sequence  of EEGTrial instances.
    #     """
    #     self.identifier = identifier 
    #     self.trials = trials

    def __init__(self, *, identifier: str, filename: str=None, trials: Sequence[EEGTrial]=None):
        """
        Initialize directly from the filename.
        """
        # Check preconditions
        if not filename and not trials:
            print("Error: Either filename or trials must not be None.")
            return
        elif filename and trials:
            print("Error: filename and trials cannot both be non-None.")
            return
        
        if trials:
            self.trials = trials
        elif filename:
            self.trials = [] # Initialize to empty list 
            data, times, labels = self.extract_from_file(filename)

            for trial_index in range(len(data)):
                trial = data[trial_index]
                self.trials.append(EEGTrial(data=trial, 
                                            times=times, 
                                            label=labels[trial_index], 
                                            trial_index=trial_index))

        self.identifier = identifier
            
    def extract_from_file(self, filename, data_extractor=None, times_extractor=None, labels_extractor=None):
        """
        Returns the data, times, and labels using the given filename. The structure of returned data
        is found in the documentation for the `__array__` method. 
        """
        raw_data = read_mat(filename)

        data = raw_data["ffr_nodss"] if not data_extractor else data_extractor(raw_data)
        times = raw_data["time"] if not times_extractor else times_extractor(raw_data)
        labels = raw_data["#subsystem#"]["MCOS"][3] if not labels_extractor else labels_extractor(raw_data)

        return data, times, labels

    def __array__(self):
        """
        Converts the array of trials into a numpy array.
        We have individual trials along the 1st dimension,
            channels along the 2nd dimension,
            and datapoints for the channel of the trial along the 3rd dimension.

            arr[a][b][c] is the c-th datapoint for the b-th channel for the a-th trial
        """
        return np.stack([trial.data for trial in self.trials])
    
    def to_subaveraged(self, n):
        """
        Returns a subaveraged vertion of this EEGData instance. 
        """
        grouped_trials = {}
        for trial in self.trials:
            trial_label = trial.label
            if trial_label in grouped_trials:
                grouped_trials[trial_label].append(trial)
            else:
                grouped_trials[trial_label] = [trial]
        
        sub_averages = []
        for _, trial_subgroup in grouped_trials.items():
            for i in range(0, len(trial_subgroup), n):
                group = trial_subgroup[i:i+n]
                sub_averages.append(AveragedEEGTrial(trials=group, trial_index=len(sub_averages)))
        
        return EEGData(identifier="test", trials=sub_averages)


if __name__ == "__main__":
    filename = "../../../trial-classification/data/4T1002.mat"
    something = EEGData(identifier="test", filename=filename)
    print(np.array(something)[0][0])



    