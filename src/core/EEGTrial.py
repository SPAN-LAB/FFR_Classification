from abc import ABC, abstractmethod
from numpy import typing as npt

class EEGTrialInterface(ABC):
    data: npt.NDArray # Currently a 1D array 
    trial_index: int
    timestamps: npt.NDArray
    raw_label: any
    mapped_label: any
    prediction: any

    @abstractmethod
    def trim_by_index(self, start_index: int, end_index: int): ... 

    @abstractmethod
    def trim_by_timestamp(self, start_time: float, end_time: float): ... 

class EEGTrial(EEGTrialInterface):
    def __init__(
        self,
        data, 
        trial_index,
        timestamps,
        raw_label,
        mapped_label=None,
        prediction=None
    ):
        self.data = data
        self.trial_index = trial_index
        self.timestamps = timestamps
        self.raw_label = raw_label
        self.mapped_label = mapped_label
        self.prediction = prediction

    def trim_by_index(self, start_index: int, end_index: int):
        self.data = self.data[start_index: end_index + 1]
        self.timestamps = self.timestamps[start_index: end_index + 1]

    def trim_by_timestamp(self, start_time: float, end_time: float):
        start_index = -1
        end_index = -1

        for i, v in enumerate(self.timestamps):
            if start_time <= v <= end_time:
                if start_index == -1:
                    start_index = i
                end_index = i

        self.trim_by_index(start_index, end_index)