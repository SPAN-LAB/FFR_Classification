from numpy import typing as npt
import numpy as np

class EEGTrialInterface:
    data: npt.NDArray # Currently a 1D array 
    trial_index: int
    timestamps: npt.NDArray
    raw_label: any
    mapped_label: any
    prediction: any
    prediction_distribution: dict[any, float]

    @property
    def label(self): 
        raise NotImplementedError("Implement this method.")

    def set_label_preference(self, pref: str | None):
        """
        There are 3 different preference types:
            - "raw" : when the `` label`` property should return the raw label
            - "mapped" : when the ``label`` property should return the mapped label
            - None : when the ``label`` property should default to the mapped label but
                     use the raw label if the mapped one is None
        """
        raise NotImplementedError("Implement this method.")
    
    def trim_by_index(self, start_index: int, end_index: int): 
        raise NotImplementedError("Implement this method.")
    
    def trim_by_timestamp(self, start_time: float, end_time: float): 
        raise NotImplementedError("Implement this method.")

    def __len__(self):
        raise NotImplementedError("Implement this method.")


class EEGTrial(EEGTrialInterface):
    def __init__(
        self,
        data,
        trial_index,
        timestamps,
        raw_label,
        mapped_label=None,
        prediction=None,
        prediction_distribution=None,
    ):
        self.data = data
        self.trial_index = trial_index
        self.timestamps = timestamps
        self.raw_label = raw_label
        self.mapped_label = mapped_label
        self.prediction = prediction
        self.prediction_distribution = prediction_distribution

        # Use default label-grabbing behavior
        self._label_preference = None

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

    def set_label_preference(self, pref: str | None = None):
        """
        "raw", "mapped", or None
        """
        self._label_preference = pref

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
