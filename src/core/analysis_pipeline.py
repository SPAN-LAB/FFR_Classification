from __future__ import annotations

import os
from copy import deepcopy

from ..printing import print, printl

from .eeg_subject import EEGSubject
from .eeg_trial import EEGTrial
from ..models.utils import ModelInterface
from ..models.utils import find_model
from .utils import detail, undetailed, gui_private, details


class AnalysisPipeline:
    def __init__(self):
        """
        Initialize the pipeline container with an empty subject list.
        """
        self.subjects: list[EEGSubject] = []
        self.models: list[ModelInterface] = []

    @undetailed()
    def save(self, *, to: PipelineState) -> AnalysisPipeline:
        """
        Captures the state of this instance in the provided instance. 
        
        Parameters
        ----------
        to : PipelineState
            the PipelineState instance the variables of this instance should be deeply copied to
        
        Returns
        -------
        This object (NOT the one this object's state was saved to)
        """
        if not isinstance(to, PipelineState):
            raise ValueError("Warning: the argument is invalid.")
        copy = self.deepcopy()
        to.subjects = copy.subjects
        to.models = copy.models
        return self
        
    @undetailed()
    def deepcopy(self) -> AnalysisPipeline:
        """
        Creates a deep copy of this object and returns a reference to it. 
        
        Returns
        -------
        The reference to the deep copy of this object.
        """
        copy = PipelineState()
        copy.subjects = deepcopy(self.subjects)
        copy.models = deepcopy(self.models)
        return copy

    # MARK: IO

    @gui_private()
    def load_subjects(self, path: str | list[str]) -> AnalysisPipeline:
        """
        Using either a file path or directory path, uses found .mat files to instantiate EEGSubject
        instances and adds them to this object's subjects list. 
        
        Parameters
        ----------
        path : str | list[str]
            A file path or directory path. If a file path is provided, its extension must be .mat.
            If a directory path is provided, then any and every .mat file in it is loaded.
            
        Returns
        -------
        A reference to this object.
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

    @detail(details.map_labels_detail)
    def map_labels(self, rule_csv: str) -> AnalysisPipeline:
        """
        Maps the labels of all subjects' EEGTrials using the contents of the file pointed to by the 
        provided file path. 
        
        Example: 
            If the contents of rule_csv are:
                1, 1, 2, 3, 4
                2, 5, 6, 7, 8
                3, 9, 10,11,12
            Then for all EEGTrial instances across all subjects this object stores, if their 
            raw_label is 1, 2, 3, or 4, then their mapped_label is set to 1; if their raw_label is 
            5, 6, 7, or 8, then their mapped_label is set to 2. 
        
        Parameters
        ----------
        rule_csv : str 
            Path to the CSV file containing the mapping rule. 
        
        Returns
        -------
        A refernece to this object
        """
        for subject in self.subjects:
            subject.map_trial_labels(rule_csv)
        print("map_labels : done")
        return self

    @detail(details.trim_by_timestamp_detail)
    def trim_by_timestamp(self, start_time: float, end_time: float) -> AnalysisPipeline:
        """
        Trims the data contained within all EEGTrials across all subjects. A datapoint is kept 
        iff its corresponding timestamp falls between start_time and end_time (inclusive).
        
        Parameters
        ----------
        start_time : float 
            The minimum timestamp of a datapoint.
        end_time : float 
            The maximum timestamp of a datapoint. 
            
        Returns
        -------
        A reference to this object.
        """
        for subject in self.subjects:
            subject.trim_by_timestamp(start_time, end_time)
        print("trim_by_timestamp : done")
        return self

    @detail(details.trim_by_index_detail)
    def trim_by_index(self, start_index: int, end_index: int) -> AnalysisPipeline:
        """
        TODO @Kevin
        """
        for subject in self.subjects:
            subject.trim_by_index(start_index, end_index)
        print("trim_by_index : done")
        return self

    @detail(details.subaverage_detail)
    def subaverage(self, size: int = 5) -> AnalysisPipeline:
        """
        Subaverages the trials of each subject. Subaveraging is performed by replacing `size` trials
        with their average. This may improve the signal-to-noise ratio of your data. If the number 
        of remaining trials is less than `size`, those trials are not included in the subaveraged 
        trials.
        
        Parameters
        ----------
        size : int 
            The number of trials to combine into one subaveraging.
        
        Returns
        -------
        A reference to this object.
        """
        for subject in self.subjects:
            subject.subaverage(size)
        print(f"subaverage ({size}) : done")
        return self

    @detail(details.fold_detail)
    def fold(self, num_folds: int = 5) -> AnalysisPipeline:
        """
        Folds the trials of each subject. That is, the trials of each subject are split across 
        `num_folds` groups in a stratified manner. This populates each subject's `folds` attribute 
        with `num_folds` elements, each a list containing the `EEGTrial`s of the individual fold.
        
        Paramters
        ---------
        num_folds : int 
            The number of groups to split each subject's trials into.
        
        Returns 
        -------
        The refernece to this object.
        """
        for subject in self.subjects:
            subject.fold(num_folds)
        print("fold : done")
        return self

    # MARK: ML functions

    @detail(details.evaluate_model_detail)
    def evaluate_model(self, model_name: str, training_options: dict[str, any]) -> AnalysisPipeline:
        """
        TODO @Kevin
        """
        concrete_model = find_model(model_name)
        for subject in self.subjects:
            # Construct the model
            model = concrete_model(training_options)
            model.set_subject(subject)

            # Evaluate it
            try:
                accuracy = model.evaluate()
                print(f"Accuracy on {subject.name}: {accuracy}")
                self.models.append(model)
            except Exception as e:
                print(f"⚠️  Error evaluating {subject.name}: {e}")
            
            # accuracy = model.evaluate()
            # print(f"Accuracy on {subject.name}: {accuracy}")
            # self.models.append(model)
        
        return self

    @detail(details.train_model_detail)
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

    @detail(details.infer_on_model_detail)
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
