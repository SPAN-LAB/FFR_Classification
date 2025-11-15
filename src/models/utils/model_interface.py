from abc import ABC, abstractmethod

from ...core import EEGSubject
from ...core import EEGTrial


class ModelInterface(ABC):
    """
    Abstract class representing any model used for FFR classification.
    """

    subject: EEGSubject

    @abstractmethod
    def __init__(self):
        """
        TODO @Kevin figure this out
        """
        ...

    def set_subject(self, subject: EEGSubject):
        """
        Binds this model class to exactly one subject.
        Call this before calling any other methods in this class.

        :param subject: the subject to set this ``ModelInterface`` instance with
        """
        self.subject = subject

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluates the accuracy of the model using cross-validation in the following steps for the subject that was passed into this object
        1. The subject data is split into folds (see 'EEGSubject.Fold' and ffr_prep.make_folds())
        2. For `i` in 1 through n where n = number of folds:
            - Let the test set be (`EEGSubject.trials`) - (trials in fold `i`)
            - Let the train set be (`EEGSubject.trials`) - (trials NOT in fold `i`)
            - Train the model on the train set's `EEGTrial` instances
            - Use that model to create predictions on the test set's `EEGTrial` instances
        3. After training on all folds, 1 prediction is made for every `EEGTrial` of the subject
        4. We check each prediction against the actual label to obtain the accuracy of the model.
        This is the accuracy of the model trained on this `EEGSubject`

        :returns: Accuracy as a float for a model on one subject's data
        """
        ...

    @abstractmethod
    def train(self):
        """
        TODO @Kevin update

        Creates a model trained on the subject this `ModelInterface` instance has been set with.
        Saves this model in ONXX format to the specified output path.

        :param output_path: the path to the location the ONXX model gets written to
        """
        ...

    @abstractmethod
    def infer(self, trials: list[EEGTrial]):
        """
        Makes exactly 1 prediction on each trial in `trials`. Sets `EEGTrial.prediction` attribute
        of each trial.

        :param trials: a list of the EEGTrials
        """
        ...

