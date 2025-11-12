from abc import ABC, abstractmethod
from ..core.eeg_subject import EEGSubject
from ..core.eeg_trial import EEGTrial
from typing import Any, Optional


class ModelInterface(ABC):
    def __init__(self):
        """
        Initializes a model with no subject bound to it.
        Call set_subject() after initializing.
        """
        self.subject: Optional[EEGSubject] = (
            None  # NOTE: to be set by set_subject() method
        )

    def set_subject(self, subject: EEGSubject):
        """
        Binds this model class to exactly one subject.
        Call this before calling any other methods in this class.
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
    def train(self, output_path: str): ...

    @abstractmethod
    def infer(self, trials: list[EEGTrial]): ...


class NeuralNetworkInterface(ModelInterface, ABC):
    """
    Extends ModelInterface to add neural network specific behavior
    The following methods are what need to be defined in addition
    to the methods specified in ModelInterface
    """

    def __init__(self, hyperparameters: dict[str, Any]):
        """
        Adds hyperparameters to constructor
        hyperparameters: dictionary that defines the hyperparameters the model will use (such as num of epochs, stopping criteria, learning rate, etc...)
        """
        super().__init__()
        self.hyperparameters = hyperparameters
        self.model: Any | None = None  # NOTE: to be set by the build() method

    @abstractmethod
    def build(self) -> None:
        """
        Construct the architecture of your model and store it in self.model
        """
        ...
