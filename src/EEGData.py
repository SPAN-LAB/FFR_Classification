
from enum import Enum, auto

class EEGData:
    class DataState(Enum):
        SHUFFLED = auto() # Indicates that the data has been shuffled 
        PLAIN = auto() # Indicates that no transformations have been applied to the data

    def __init__(self, data, times, labels):
        self.data = data
        self.times = times
        self.labels = labels

        self.state = self.DataState.PLAIN # Default state when nothing has been done yet

    def transpose_data(self):
        """Modify self.data to be trials-by-time using the prebuilt transpose function.
        Detects current orientation using the length of self.times.
        """
        # from preprocessing_functions import transpose_ffr as _transpose_ffr
        print(f"initial shape: {self.data.shape}")
        # if self.data is None or self.times is None:
        #     raise ValueError("EEGData.transpose_ffr requires non-empty data and times")

        # rows, cols = self.data.shape
        # try:
        #     num_time = self.times.size
        # except Exception:
        #     # Fallback if times is a list
        #     num_time = len(self.times)

        # # Ensure input to transpose_ffr is time-by-trial
        # if rows == num_time:
        #     all_ffr = self.data  # time x trial
        #     num_trials = cols
        # elif cols == num_time:
        #     all_ffr = self.data.T  # make time x trial
        #     num_trials = rows
        # else:
        #     # Assume rows are time as a fallback
        #     all_ffr = self.data
        #     num_trials = cols

        # transposed = _transpose_ffr(all_ffr=all_ffr, num_trials=num_trials, all_time=self.times)
        # self.data = transposed

        self.data = self.data.T
        print(f"final shape: {self.data.shape}")