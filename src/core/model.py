from abc import ABC, abstractmethod
from .eeg_subject import EEGSubject
from .eeg_trial import EEGTrial

class ModelInterface(ABC):
    subjects: list[EEGSubject]

    @abstractmethod
    def evaluate(self) -> list[float]:
        """
        Evaluates the accuracy of the model using cross-validation in the following steps for EACH 
        subject in `self.subjects`:
        1. The subject data is split into folds (see `EEGSubject.fold`)
        2. For `i` in 1 through n where n = number of folds:
            - Let the test set be (`EEGSubject.trials`) - (trials in fold `i`)
            - Let the train set be (`EEGSubject.trials`) - (trials NOT in fold `i`)
            - Train the model on the train set's `EEGTrial` instances
            - Use that model to create predictions on the test set's `EEGTrial` instances
        3. After training on all folds, 1 prediction is made for every `EEGTrial` of the subject
        4. We check each prediction against the actual label to obtain the accuracy of the model.
        This is the accuracy of the model trained on this `EEGSubject`

        :returns: list of accuracies of the model trained on each subject
        """
        ...

    @abstractmethod
    def train(self, output_path: str): ...

    @abstractmethod
    def infer(self, trials: list[EEGTrial]): ... 

class NeuralNetwork(ModelInterface): 
    def __init__(self, hyperparameters: dict[str, any]):
        self.subjects = []
        self.model = None
        self.hyperparameters = None

        # Setup hyperparamters
        self.set_hyperparameters(hyperparameters)
    
    def add_subjects(self, subjects: list[EEGSubject]):
        self.subjects.extend(subjects)

    def set_hyperparameters(self, hyperparameters: dict[str, any]):
        self.hyperparameters = hyperparameters

    def load(self, name: str):
        """
        Loads the specified model from /mmodels

        :param str name: the name of the model to load
        """
        # self.model = ???
        pass

    def evaluate(self) -> float:
        """
        Evaluates the performance of the loaded model using cross validation.
        Returns a float between 0 and 1 representing the accuracy of the model.
        """
        use_gpu = self.hyperparameters["use_gpu"]
        num_epochs = self.hyperparameters["num_epochs"]
        learning_rate = self.hyperparameters["learning_rate"]
        stopping_criteria = self.hyperparameters["stopping_criteria"]

        return 0.0

    def train(self, output_path: str):
        """
        Creates a
        """
        pass