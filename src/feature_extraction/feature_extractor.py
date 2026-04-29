from ..core import EEGSubject, EEGTrial
from numpy import typing as npt

class FeatureExtractor:
    def _get_features(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """
        data: an array of numbers 
        returns an array of numbers
        """
        raise NotImplementedError("This methods needs to be implemented.")
    
    def transform(self, data: EEGTrial | list[EEGTrial] | EEGSubject):
        # Convert input into a list of EEGTrial instances
        if isinstance(data, EEGTrial):
            data = [data]
        if isinstance(data, EEGSubject):
            data = data.trials

        for trial in data:
            trial.data = self._get_features(trial.data)
        
class FeatureGenerator(FeatureExtractor):
    def _generate_features(self, trial_data: list[npt.ArrayLike]):
        """
        Creates features using a list of EEGTrial instances
        """
        raise NotImplementedError("This method needs to be implemented.")
    
    def generate_features(self, trials: list[EEGTrial] | EEGSubject):
        if isinstance(trials, EEGSubject):
            trials = trials.trials
        trial_data = [trial.data for trial in trials]
        self._generate_features(trial_data)

class ExampleFeatureExtractor(FeatureExtractor):
    def _get_features(self, data: npt.ArrayLike) -> npt.ArrayLike:
        for i in range(len(data)):
            data[i] = data[i] + 1
        return data

class ExampleFeatureGenerator(FeatureGenerator):
    def __init__(self):
        self.shifter = 0
    
    def _get_features(self, data: npt.ArrayLike) -> npt.ArrayLike:
        for i in range(len(data)):
            data[i] = data[i] + self.shifter
        return data
    
    def _generate_features(self, trial_data: list[npt.ArrayLike]):
        if len(trial_data) % 2 == 0:
            self.shifter = 1
        else:
            self.shifter = -1
