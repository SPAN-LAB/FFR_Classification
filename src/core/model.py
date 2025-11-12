from abc import ABC, abstractmethod
from .eeg_subject import EEGSubject
from .eeg_trial import EEGTrial

class ModelInterface(ABC):
    subject: EEGSubject

    def set_subject(self, subject: EEGSubject): 
        self.subject = subject

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluates the accuracy of the model using cross-validation in the following steps:
        1. The subject data is split into folds (see `EEGSubject.fold`)
        2. For `i` in 1 through n where n = number of folds:
            - Let the test set be (`EEGSubject.trials`) - (trials in fold `i`)
            - Let the train set be (`EEGSubject.trials`) - (trials NOT in fold `i`)
            - Train the model on the train set's `EEGTrial` instances
            - Use that model to create predictions on the test set's `EEGTrial` instances
        3. After training on all folds, 1 prediction is made for every `EEGTrial` of the subject
        4. Each prediction is checked against the actual label to obtain the accuracy of the model.
        This is the accuracy of the model trained on this `EEGSubject`

        :returns: accuracy of the model train on its subject data 
        """
        ...

    @abstractmethod
    def train(self, output_path: str): ...

    @abstractmethod
    def infer(self, trials: list[EEGTrial]): ... 

class NeuralNetwork(ModelInterface): 
    def __init__(self, hyperparameters: dict[str, any]):
        self.subject = None
        self.model = None
        self.hyperparameters = None

        # Setup hyperparamters
        self.set_hyperparameters(hyperparameters)

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

    def infer(self, trials: list[EEGTrial]):
        """
        
        """
        pass