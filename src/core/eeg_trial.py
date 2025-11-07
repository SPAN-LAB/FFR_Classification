from abc import ABC, abstractmethod
from numpy import typing as npt
import numpy as np

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
        start = int(np.searchsorted(self.timestamps, start_time, side="left"))
        end = int(np.searchsorted(self.timestamps, end_time, side="right"))
        self.timestamps = self.timestamps[start:end]
        self.data = self.data[start:end]