from __future__ import annotations
from .eeg_subject import EEGSubject
from .eeg_trial import EEGTrial
import os

class PipelineState:
    def __init__(self):
        """
        Initialize the pipeline container with an empty subject list.
        """
        self.subjects: list[EEGSubject] = []
    
    ### IO ###

    def load_subjects(self, folder_path: str) -> PipelineState:
        """
        Load every `.mat` file in a directory as an `EEGSubject`.

        :param str folder_path: The directory containing the `.mat` files.
        """
        for file in os.listdir(folder_path):
            if file.endswith(".mat"):
                subject = EEGSubject.init_from_filepath(os.path.join(folder_path, file))
                self.subjects.append(subject)
        return self

    ### Pre-training processing functions ### 

    def map_labels(self, rule_csv: str) -> PipelineState:
        """
        Sets the labels for all trials according to the provided file.

        :param str rule_csv: The file containing the mapping rule.
        """
        for subject in self.subjects: 
            subject.map_trial_labels(rule_csv)
        return self

    def trim_by_timestamp(self, start_time: float, end_time: float) -> PipelineState: 
        for subject in self.subjects: 
            subject.trim_by_timestamp(start_time, end_time)
        return self

    def trim_by_index(self, start_index: int, end_index: int) -> PipelineState: 
        for subject in self.subjects: 
            subject.trim_by_index(start_index, end_index)
        return self

    def subaverage(self, size: int = 5) -> PipelineState:
        for subject in self.subjects: 
            subject.subaverage(size)
        return self

    def fold(self, num_folds: int = 5) -> PipelineState:
        for subject in self.subjects: 
            subject.fold(num_folds)
        return self

    ### ML functions ### 

    def evaluate_model(self, model_name: str) -> PipelineState:
        pass

    def train_model(self, output_path: str) -> PipelineState: 
        """
        Train on 100% of the data 
        """
        pass 

    def infer_on_model(self, model: Model, trial: EEGTrial) -> PipelineState:
        """
        
        """
        pass