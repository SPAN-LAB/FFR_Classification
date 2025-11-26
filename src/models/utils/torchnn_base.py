from abc import abstractmethod

VERBOSE = True

from src.core import ffr_proc
from ...core import FFRPrep
from ...core.ffr_proc import get_accuracy
from ...core import EEGSubject
from ...core import EEGTrial  # This isn't being used now but will be when ``infer`` is implemented
from .model_interface import ModelInterface

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import json

class EEGDataset(Dataset):
    def __init__(self, trials: list[EEGTrial]):
        self.trials = trials

    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, index):
        """
        Returns a tuple.

        The first element is numpy array of the datapoints captured for the single trial.
        
        The second element is the label associated with the first element's data.

        The third element is the index associated with the data.
        """
        trial_data = torch.tensor(self.trials[index].data, dtype=torch.float32)
        trial_label = torch.tensor(self.trials[index].enumerated_label, dtype=torch.long)
        trial_index = self.trials[index].trial_index
        return trial_data, trial_label, trial_index

class TorchNNBase(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        # Initialies subject and training_options attributes
        super().__init__(training_options)

        self.model: nn.Module | None = None
        self.device = None

        # Automatically attempt to use the GPU
        self.set_device()

    def set_device(self, use_gpu: bool = True):
        """
        Searches for a compatible GPU device if ``use_gpu`` is True. 
        If one isn't found, or if ``use_gpu`` is False, uses the CPU instead. 
        """
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Set to MPS")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def build(self):
        raise NotImplementedError("This method needs to be implemented")

    def trials_loader(
        self,
        trials: list[EEGTrial], 
        batch_size: int,
        shuffle: bool = True,
        pin_memory: bool = True
    ) -> DataLoader:
        return DataLoader(
            EEGDataset(trials),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory
        )
    
    def create_loaders(
        self,
        folds: list[list[EEGTrial]],
        withheld_fold_index: int,
        validate_ratio: float,
        batch_size: int,
        shuffle: bool = True,
        pin_memory: bool = True
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        # The ``EEGTrial`` instances used for training
        train_trials = []
        for i, fold in enumerate(folds):
            if i == withheld_fold_index:
                continue
            for trial in fold:
                train_trials.append(trial)

        # The ``EEGTrial`` instances used for validation 
        # TODO
        validate_trials = folds[0][0]

        # The ``EEGTrial`` instances used for testing
        test_trials = folds[withheld_fold_index]
        
        train_loader = self.trials_loader(
            trials=train_trials, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=pin_memory
        )
        
        validate_loader = self.trials_loader(
            trials=validate_trials, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=pin_memory
        )

        test_loader = self.trials_loader(
            trials=test_trials, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=pin_memory
        )

        return train_loader, validate_loader, test_loader

    def evaluate(self) -> float:
        """
        Uses K-fold CV to train and test on EEGSubject data
        and returns overall accuracy as a float

        Preconditions:
            -self.subject != None
            -self.model != None
            -self.subject.folds != None
        """

        batch_size = self.training_options["batch_size"]
        learning_rate = self.training_options["learning_rate"]
        num_epochs = self.training_options["num_epochs"]
        weight_decay = self.training_options["weight_decay"]

        for i in range(len(self.subject.folds)):

            self.build()
            self.model = self.model.to(self.device)

            train_loader, validate_loader, test_loader = self.create_loaders(
                folds=self.subject.folds,
                withheld_fold_index=i,
                validate_ratio=0.0,
                batch_size=batch_size
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            for epoch in range(num_epochs):

                trial_count = 0
                correct_count = 0
                
                self.model.train()
                for data_batch, labels_batch, indices_batch in train_loader:

                    data_batch = data_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    indices_batch = indices_batch.to(self.device)
                    
                    # Zero-out the gradients to prevent accumulating them from previous iterations
                    optimizer.zero_grad()

                    # Forward pass through the model
                    logits = self.model(data_batch)

                    # Calculate the loss 
                    loss = criterion(logits, labels_batch)

                    # Compute gradients for every parameter
                    loss.backward()

                    # Update each parameter
                    optimizer.step()

                    # Update the predictions 
                    for k, trial_index in enumerate(indices_batch):
                        prediction = logits[k].argmax().item()
                        
                        trial_count += 1
                        
                        if prediction == self.subject.trials[trial_index].enumerated_label:
                            correct_count +=1 
                
                print(f"Accuracy on epoch {epoch + 1} on subject {self.subject.name}: {(100 * correct_count / trial_count):.4f}")

                    
            self.model.eval()
            with torch.no_grad(): # Turn off gradient-tracking

                test_trial_count = 0
                test_correct_count = 0

                for data_batch, labels_batch, indices_batch in test_loader:
                    
                    data_batch = data_batch.to(self.device)
                    # labels_batch = labels_batch.to(self.device)
                    # indices_batch = indices_batch.to(self.device)
                    
                    # Forward pass through the model
                    logits = self.model(data_batch)

                    # Update the predictions 
                    for k, trial_index in enumerate(indices_batch):
                        prediction = logits[k].argmax().item()
                        self.subject.trials[trial_index].set_prediction(prediction)

                        test_trial_count += 1
                        
                        if prediction == self.subject.trials[trial_index].enumerated_label:
                            test_correct_count += 1

                print(f"Test accuracy for fold {i + 1}: {(100 * test_correct_count / test_trial_count):.4f}")

        return get_accuracy(self.subject)


    def train(self):
        """
        To be implemented later
        """
        pass

    def infer(self, trials: list[EEGTrial]):
        """
        To be implemented later
        """
        pass
