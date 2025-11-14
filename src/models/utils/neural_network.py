from ...core import EEGSubject, EEGTrial
from .model import ModelInterface
from ...core import FFRPrep

import torch

class NeuralNetwork(ModelInterface): 
    """
    Represents parent class for a PyTorch neural network, providing all methods needed to use any 
    PyTorch neural network. 
    """

    def __init__(self, hyperparameters: dict[str, any]):
        self.subject: EEGSubject | None = None
        self.model = None
        self.device = None
        self.hyperparameters: dict | None = None

        # Setup hyperparamters
        self.set_hyperparameters(hyperparameters)

        # Setup the device automatically
        self.set_device()

    def set_hyperparameters(self, hyperparameters: dict[str, any]):
        """
        Setter for this instance's hyperparameters.
        """
        self.hyperparameters = hyperparameters

    def set_device(self, use_gpu: bool = True):
        """
        Setter for the device to run the models on.
        """
        # Determine the device 
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            use_gpu
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        self.device = device
        if self.model is not None:
            self.model.to(device)

    def evaluate(self) -> float:
        """
        Evaluates the performance of the loaded model using cross validation.
        Returns a float between 0 and 1 representing the accuracy of the model.
        """
        # training_options
        num_epochs = self.hyperparameters["num_epochs"]
        learning_rate = self.hyperparameters["learning_rate"]
        batch_size = self.hyperparameters["batch_size"]
        stop_threshold: float | None = self.hyperparameters["stop_threshold"] # None if no threshold

        # Note that we assume that the subject data has already been split into folds.
        # This means you don't need to fold it again. 
        # To access the folds, simply do: 
        # 
        #     folds = subject.folds
        # 
        # ...to get the folds.

        folds = self.subject.folds
        for i, fold in enumerate(folds):
            
            # Create the model

            # Create the train and validation dataloaders using all but the i-th fold

            # Create the test dataloader 

            # Setting up the criterion and optimizer 

            for ep in range(num_epochs):
                
                # Forward pass through the model for training

                # Validate the model, exiting if it meets the stopping threshold

                pass
        
            # Test the model, populating the `EEGTrial` instances of the i-th fold with predictions
            

        # NOTE: Nike said that the logic for finding the accuracy should be pre-written, which I
        # agree with, since it doesn't depend on the model.

        return FFRPrep.get_accuracy(self.subject)


    def train(self, output_path: str):
        """
        Creates a
        """
        pass

    def infer(self, trials: list[EEGTrial]):
        """
        
        """
        pass