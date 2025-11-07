from abc import ABC, abstractmethod
from .eeg_subject import EEGSubject
from .eeg_trial import EEGTrial

class ModelInterface(ABC):
    subjects: list[EEGSubject]

    @abstractmethod
    def evaluate(self) -> float: ...

    @abstractmethod
    def train(self, output_path: str): ...

    @abstractmethod
    def infer(self, trials: list[EEGTrial]): ... 

class NeuralNetwork(ModelInterface): 
    def __init__(self, hyperparameters: dict[str, int]):
        self.subjects = []
        self.model = None
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