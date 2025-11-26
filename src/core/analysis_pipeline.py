from __future__ import annotations
from .eeg_subject import EEGSubject
from .eeg_trial import EEGTrial

from .utils import FunctionDetail as FD
from .utils import ArgumentDetail as AD
from .utils import Selection 
from .utils.details import *

from ..models.utils import ModelInterface
from ..models.utils import find_model
import os
from copy import deepcopy

class AnalysisPipeline:
    def __init__(self):
        """
        Initialize the pipeline container with an empty subject list.
        """
        self.subjects: list[EEGSubject] = []
        self.models: list[ModelInterface] = []
    
    @undetailed()
    def save(self, *, to: PipelineState) -> AnalysisPipeline:
        to.subjects = deepcopy(self.subjects)
        to.models = deepcopy(self.models)
        return self

    # MARK: IO

    @gui_private()
    def load_subjects(self, path: str | list[str]) -> AnalysisPipeline:
        """
        Load every `.mat` file in a directory or list of filepaths as `EEGSubject` instances.

        :param path: Either a string path (file or directory) or a list of file paths.
        """

        def load_subjects_helper(filepath: str, check_extension: bool = True):
            if check_extension and not filepath.endswith(".mat"):
                raise ValueError(f"File does not end with .mat: {filepath}")
            subject = EEGSubject.init_from_filepath(filepath)
            print(f"load_subjects : Subject loaded from {filepath}")
            self.subjects.append(subject)

        if type(path) is str:
            
            if os.path.isfile(path):
                load_subjects_helper(path)
            elif os.path.isdir(path):
                files = []
                for file in os.listdir(path):
                    if file.endswith(".mat"):
                        files.append(file)

                if len(files) == 0:
                    print(f"Warning: No .mat files found in directory: {path}")
                for file in files:
                    filepath = os.path.join(path, file)
                    load_subjects_helper(filepath, check_extension=False)
            else:
                raise ValueError(f"Path does not exist: {path}")
        elif type(path) is list:
            for filepath in path:
                load_subjects_helper(filepath)
        else:
            raise ValueError("Unrecognized input: path must be a string or list of strings")
        
        return self

    # MARK: Pre-training processing functions

    @detail(map_labels_detail)
    def map_labels(self, rule_csv: str) -> AnalysisPipeline:
        """
        Sets the labels for all trials according to the provided file.

        :param str rule_csv: The file containing the mapping rule.
        """
        for subject in self.subjects: 
            subject.map_trial_labels(rule_csv)
        print(f"map_labels : done")
        return self

    @detail(trim_by_timestamp_detail)
    def trim_by_timestamp(self, start_time: float, end_time: float) -> AnalysisPipeline: 
        for subject in self.subjects: 
            subject.trim_by_timestamp(start_time, end_time)
        print(f"trim_by_timestamp : done")
        return self

    # @detail(trim_by_index_detail)
    @undetailed()
    def trim_by_index(self, start_index: int, end_index: int) -> AnalysisPipeline: 
        for subject in self.subjects: 
            subject.trim_by_index(start_index, end_index)
        print(f"trim_by_index : done")
        return self

    @detail(subaverage_detail)
    def subaverage(self, size: int = 5) -> AnalysisPipeline:
        for subject in self.subjects: 
            subject.subaverage(size)
        print(f"subaverage : done")
        return self

    @detail(fold_detail)
    def fold(self, num_folds: int = 5) -> AnalysisPipeline:
        for subject in self.subjects: 
            subject.fold(num_folds)
        print(f"fold : done")
        return self

    # MARK: ML functions

    @detail(evaluate_model_detail_2)
    def evaluate_model(self, model_name: str, training_options: dict[str, any]) -> AnalysisPipeline:
        concrete_model = find_model(model_name)
        for subject in self.subjects:
            # Construct the model 
            model = concrete_model(training_options)
            model.set_subject(subject)

            # Evaluate it
            print(f"Accuracy on {subject.name}: {model.evaluate()}")
            self.models.append(model)

    @detail(train_model_detail)
    def train_model(
        self, 
        model_name: str, 
        hyperparameters: dict[str, any], 
        output_path: str
    ) -> AnalysisPipeline: 
        """
        TODO
        """
        concrete_model = find_model(model_name)
        for subject in self.subjects:
            # Construct the model
            model = concrete_model(hyperparameters)
            model.set_subject(subject)

            # Train and save to disk 
            model.train(output_path)
        
        return self

    @detail(infer_on_model_detail)
    def infer_on_model(self, path_to_model: str, trial: EEGTrial) -> AnalysisPipeline:
        """
        TODO
        """
        # TODO
        return self
    
# Type alias for ``AnalysisPipeline`` for more expressive use.
# When the ``AnalysisPipeline`` is isolated in the middle, it makes semantic sense for it to be 
# called a ``PipelineState``
PipelineState = AnalysisPipeline