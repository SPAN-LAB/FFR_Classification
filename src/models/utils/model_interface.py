"""
SPAN Lab - FFR Classification

Filename: model_interface.py
Author(s): Kevin Chen
Description: An interface that all ML models must conform to for compatability with 
    AnalysisPipeline.
"""


from abc import ABC, abstractmethod

from copy import deepcopy
from pathlib import Path
import pickle

from ...core import EEGSubject
from ...core import EEGTrial
from ...time import TimeKeeper

from ...configurations import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


class ModelInterface(ABC):
    """
    Abstract class representing any model used for FFR classification.
    """

    subject: EEGSubject
    
    def __init__(self, training_options: dict[str, any]):
        self.subject = None
        self.training_options = training_options
    
    def set_subject(self, subject: EEGSubject):
        self.subject = subject
    
    def set_training_options(self, training_options: dict[str, any]):
        self.training_options = training_options

    # MARK: Core functions
    
    def _core_infer(self, *,
        trials: list[EEGTrial],
        batch_Size: int
    ) -> list[dict[int, float]]:
        raise NotImplementedError("This method need to be implemented.")
    
    def _core_train(self, *, 
        trials: list[EEGTrial],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float
    ):
        raise NotImplementedError("This method need to be implemented.")

    def train(self, *, 
        trials: list[EEGTrial] = [], 
        pickle_to: str | Path | None = None,
        overwrite: bool = True
    ):
        
        if pickle_to is not None:
            if isinstance(pickle_to, str):
                pickle_to = Path(pickle_to)
            
            if pickle_to.is_dir():
                raise ValueError("Path provided is a directory")
            if not overwrite and pickle_to.exists():
                raise ValueError("File already exists at the provided location")
        
        if len(trials) == 0:
            trials = self.subject.trials
        
        model = self._core_train(trials=trials)
        
        # Save to this instance
        self.model = model
        
        # Save to disk if path is specified
        if pickle_to is not None:
            self_copy = deepcopy(self)
            self_copy.subject = None
            pickle_to.parent.mkdir(parents=True, exist_ok=True)
            with pickle_to.open("wb") as file:
                pickle.dump(self, file)
            
            print(f"Model written to {pickle_to.absolute()}")
    
    def infer(self, *, trials: list[EEGTrial]) -> float:
        
        if len(trials) == 0:
            trials = self.subject.trials
        
        batch_size = self.training_options.get("batch_size", BATCH_SIZE)
        prediction_distributions = self._core_infer(
            trials=trials, 
            batch_size=batch_size
        )
        
        for i, trial in enumerate(trials):
            trial.set_prediction_distribution(
                enumerated_prediction_distribution=prediction_distributions[i]
            )
        
        return EEGTrial.get_accuracy(trials)
    
    def _cross_validate(self, *,
        folded_trials: list[list[EEGTrial]],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float
    ) -> list[list[dict[int, float]]]:
        
        num_folds = len(folded_trials)
        folded_trial_prediction_distributions = [[] for _ in range(num_folds)]
        
        per_fold_tk = TimeKeeper()
        per_fold_tk.reset()
        per_fold_tk.start()

        for fold_i in range(num_folds):
            
            # Create the train and test lists
            train_trials = []
            test_trials = []
            for withheld_i, trial_list in enumerate(folded_trials):
                if withheld_i == fold_i:
                    test_trials += trial_list
                else:
                    train_trials += trial_list
            
            self._core_train(
                trials=train_trials,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            
            prediction_distributions = self._core_infer(
                trials=test_trials,
                batch_size=batch_size
            )
            
            folded_trial_prediction_distributions[fold_i] += prediction_distributions
        
        per_fold_tk.stop()
        print(f"All folds took {per_fold_tk.accumulated_duration} seconds.")
            
        return folded_trial_prediction_distributions
    
    def evaluate(self, *, 
        folded_trials: list[list[EEGTrial]] = []
    ) -> float:
        
        if len(folded_trials) == 0:
            if self.subject == None:
                raise ValueError("No folds were provided; self.subject is None")
            folded_trials = self.subject.folds
        
        # Set up 
        num_epochs = self.training_options.get("num_epochs", NUM_EPOCHS)
        batch_size = self.training_options.get("batch_size", BATCH_SIZE)
        learning_rate = self.training_options.get("learning_rate", LEARNING_RATE)
        weight_decay = self.training_options.get("weight_decay", WEIGHT_DECAY)
        
        folded_prediction_distributions = self._cross_validate(
            folded_trials=folded_trials,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        for i in range(len(folded_trials)):
            for j in range(len(folded_trials[i])):
                folded_trials[i][j].set_prediction_distribution(
                    enumerated_prediction_distribution=folded_prediction_distributions[i][j]
                )
        
        return EEGTrial.get_accuracy(trials=folded_trials)
