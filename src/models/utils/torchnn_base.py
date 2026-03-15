"""
SPAN Lab - FFR Classification

Filename: torchnn_base.py
Author(s): TODO @Anu
Description: TODO @Anu
"""


from abc import abstractmethod

from torch.utils.data.dataloader import DataLoader

from ...printing import print, printl, unlock
from ...time import TimeKeeper

from ...core.ffr_prep import IndexTrackedDataset
from ...core.ffr_proc import get_accuracy
from ...core import EEGSubject
from ...core import EEGTrial  # This isn't being used now but will be when ``infer`` is implemented
from .model_interface import ModelInterface

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
import json
from copy import deepcopy

import pickle
from ...configurations import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY


class TorchNNBase(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        # Initialize ``self.training_options``
        super().__init__(training_options)

        torch.manual_seed(0)

        self.model: nn.Module | None = None
        self.device = None

        self.set_device() # Automatically attempt to use the GPU

    def set_device(self, use_gpu: bool = True):
        """
        Searches for a compatible GPU device if ``use_gpu`` is True.
        If one isn't found, or if ``use_gpu`` is False, uses the CPU instead.
        """
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA for GPU computations")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS for GPU computations")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for torch computations")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for torch computations")

    def build(self):
        raise NotImplementedError("This method needs to be implemented")

    def _core_infer(self, *, 
        trials: list[EEGTrial],
        batch_size: int = BATCH_SIZE
    ) -> list[dict[int, float]]:
        
        dataloader = DataLoader(
            IndexTrackedDataset(trials=trials),
            batch_size=batch_size,
            shuffle=False
        )
        
        prediction_distributions = [None] * len(trials)
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                
                inputs = batch["x"].to(self.device)
                indices = batch["index"]
                logits = self.model(inputs)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

                for i, probability_row in enumerate(probabilities):
                    
                    index = indices[i].item()

                    prediction_distribution = {
                        enumerated_label: float(p) 
                        for enumerated_label, p in enumerate(probability_row)
                    }
                    
                    prediction_distributions[index] = prediction_distribution
        
        return prediction_distributions
    
    def _core_train(self, *, 
        trials: list[EEGTrial], 
        num_epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY
    ) -> nn.Module:
        
        # Set up 
        
        self.build()
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        train_loader = DataLoader(
            IndexTrackedDataset(trials=trials),
            batch_size=batch_size,
            shuffle=False
        )
        
        epoch_tk = TimeKeeper()
        epoch_tk.reset()
        epoch_tk.start()
        
        self.model.train()
        for epoch_i in range(num_epochs):
            
            for batch in train_loader:
                
                inputs = batch["x"].to(self.device)
                labels = batch["y"].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            epoch_tk.lap()
            print_content = (
                f"Epoch [ {epoch_i + 1}/{num_epochs} ]"
                f" | Fold time: [ {epoch_tk.accumulated_duration:.3f}s ]"
                f" | [ {epoch_tk.last_lap_duration:.3f}s / epoch ]"
            )
            printl(print_content)
        
        unlock()
        
        return self.model
        
    # def _core_cross_validate(self, *, 
    #     folded_trials: list[list[EEGTrial]],
    #     num_epochs: int = NUM_EPOCHS,
    #     batch_size: int = BATCH_SIZE,
    #     learning_rate: float = LEARNING_RATE,
    #     weight_decay: float = WEIGHT_DECAY
    # ) -> list[list[dict[int, float]]]:
        
    #     num_folds = len(folded_trials)
    #     folded_trial_prediction_distributions = [[] for _ in range(num_folds)]
        
    #     per_fold_tk = TimeKeeper()
    #     per_fold_tk.reset()
    #     per_fold_tk.start()

    #     for fold_i in range(num_folds):
            
    #         # Create the train and test lists
    #         train_trials = []
    #         test_trials = []
    #         for withheld_i, trial_list in enumerate(folded_trials):
    #             if withheld_i == fold_i:
    #                 test_trials += trial_list
    #             else:
    #                 train_trials += trial_list
            
    #         self._core_train(
    #             trials=train_trials,
    #             num_epochs=num_epochs,
    #             batch_size=batch_size,
    #             learning_rate=learning_rate,
    #             weight_decay=weight_decay
    #         )
            
    #         prediction_distributions = self._core_infer(
    #             trials=test_trials,
    #             batch_size=batch_size
    #         )
            
    #         folded_trial_prediction_distributions[fold_i] += prediction_distributions
        
    #     per_fold_tk.stop()
    #     print(f"All folds took {per_fold_tk.accumulated_duration} seconds.")
            
    #     return folded_trial_prediction_distributions

    # def evaluate(self, *, folded_trials: list[list[EEGTrial]] = []) -> float:
        
    #     if len(folded_trials) == 0:
    #         if self.subject == None:
    #             raise ValueError("No folds were provided; self.subject is None")
    #         folded_trials = self.subject.folds
        
    #     # Set up 
    #     num_epochs = self.training_options.get("num_epochs", NUM_EPOCHS)
    #     batch_size = self.training_options.get("batch_size", BATCH_SIZE)
    #     learning_rate = self.training_options.get("learning_rate", LEARNING_RATE)
    #     weight_decay = self.training_options.get("weight_decay", WEIGHT_DECAY)
        
    #     folded_prediction_distributions = self._core_cross_validate(
    #         folded_trials=folded_trials,
    #         num_epochs=num_epochs,
    #         batch_size=batch_size,
    #         learning_rate=learning_rate,
    #         weight_decay=weight_decay
    #     )
        
    #     for i in range(len(folded_trials)):
    #         for j in range(len(folded_trials[i])):
    #             folded_trials[i][j].set_prediction_distribution(
    #                 enumerated_prediction_distribution=folded_prediction_distributions[i][j]
    #             )
        
    #     return EEGTrial.get_accuracy(trials=folded_trials)

    # def train(self, *, 
    #     trials: list[EEGTrial] = [], 
    #     pickle_to: str | Path | None = None,
    #     overwrite: bool = True
    # ):
        
    #     if pickle_to is not None:
    #         if isinstance(pickle_to, str):
    #             pickle_to = Path(pickle_to)
            
    #         if pickle_to.is_dir():
    #             raise ValueError("Path provided is a directory")
    #         if not overwrite and pickle_to.exists():
    #             raise ValueError("File already exists at the provided location")
        
    #     if len(trials) == 0:
    #         trials = self.subject.trials
        
    #     model = self._core_train(trials=trials)
        
    #     # Save to this instance
    #     self.model = model
        
    #     # Save to disk if path is specified
    #     if pickle_to is not None:
    #         self_copy = deepcopy(self)
    #         self_copy.subject = None
    #         pickle_to.parent.mkdir(parents=True, exist_ok=True)
    #         with pickle_to.open("wb") as file:
    #             pickle.dump(self, file)
            
    #         print(f"Model written to {pickle_to.absolute()}")
            
    # def infer(self, *, trials: list[EEGTrial] = []) -> float:
        
    #     if len(trials) == 0:
    #         trials = self.subject.trials
        
    #     batch_size = self.training_options.get("batch_size", BATCH_SIZE)
    #     prediction_distributions = self._core_infer(
    #         trials=trials, 
    #         batch_size=batch_size
    #     )
        
    #     for i, trial in enumerate(trials):
    #         trial.set_prediction_distribution(
    #             enumerated_prediction_distribution=prediction_distributions[i]
    #         )
        
    #     return EEGTrial.get_accuracy(trials)
        
