"""
SPAN Lab - FFR Classification

Filename: model_interface.py
Author(s): Kevin Chen
Description: An interface that all ML models must conform to for compatability with 
    AnalysisPipeline.
"""


from copy import deepcopy
from pathlib import Path
import pickle

from ...core import EEGSubject
from ...core import EEGTrial
from ...time import TimeKeeper

from ...printing import print, printl, unlock

from ...constants.defaults import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


class ModelInterface:
    """
    Abstract class representing any model used for FFR classification.
    """
    
    def __init__(self, training_options: dict[str, any]):
        self.subject = None
        self.training_options = training_options
    
    # MARK: Setters
    
    def set_subject(self, subject: EEGSubject):
        self.subject = subject
    
    def set_training_options(self, training_options: dict[str, any]):
        self.training_options = training_options
    
    def reset_seed(self):
        raise NotImplementedError("This needs to be implemented!")

    # MARK: Training parameter getters

    def get_num_epochs(self) -> int:
        if not isinstance(self.training_options, dict):
            return NUM_EPOCHS
        return self.training_options.get("num_epochs", NUM_EPOCHS)

    def get_batch_size(self) -> int:
        if not isinstance(self.training_options, dict):
            return BATCH_SIZE
        return self.training_options.get("batch_size", BATCH_SIZE)
    
    def get_learning_rate(self) -> int:
        if not isinstance(self.training_options, dict):
            return LEARNING_RATE
        return self.training_options.get("learning_rate", LEARNING_RATE)
    
    def get_weight_decay(self) -> int:
        if not isinstance(self.training_options, dict):
            return WEIGHT_DECAY
        return self.training_options.get("weight_decay", WEIGHT_DECAY)
        
    # MARK: Printing
    
    def update_printed_training_status(self, content):
        printl(content)
    
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

    # MARK: Orchestration functions
    
    def infer(self, *, trials: list[EEGTrial]) -> float:
        
        if len(trials) == 0:
            trials = self.subject.trials

        prediction_distributions = self._core_infer(
            trials=trials, 
            batch_size=self.get_batch_size()
        )
        
        for i, trial in enumerate(trials):
            trial.set_prediction_distribution(
                enumerated_prediction_distribution=prediction_distributions[i]
            )
        
        return EEGTrial.get_accuracy(trials)
    
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
        
        # Training
        self.reset_seed()
        self._core_train(
            trials=trials,
            num_epochs=self.get_num_epochs(),
            batch_size=self.get_batch_size(),
            learning_rate=self.get_learning_rate(),
            weight_decay=self.get_weight_decay()
        )
        unlock()
        
        # Save to disk if path is specified
        if pickle_to is not None:
            self_copy = deepcopy(self)
            self_copy.subject = None
            pickle_to.parent.mkdir(parents=True, exist_ok=True)
            with pickle_to.open("wb") as file:
                pickle.dump(self, file)
            
            print(f"Model written to {pickle_to.absolute()}")
    
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
            
            self.reset_seed()
            self._core_train(
                trials=train_trials,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            unlock()
            
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
            if self.subject is None:
                raise ValueError("No folds were provided; self.subject is None")
            folded_trials = self.subject.folds
        
        folded_prediction_distributions = self._cross_validate(
            folded_trials=folded_trials,
            num_epochs=self.get_num_epochs(),
            batch_size=self.get_batch_size(),
            learning_rate=self.get_learning_rate(),
            weight_decay=self.get_weight_decay()
        )
        
        for i in range(len(folded_trials)):
            for j in range(len(folded_trials[i])):
                folded_trials[i][j].set_prediction_distribution(
                    enumerated_prediction_distribution=folded_prediction_distributions[i][j]
                )
        
        return EEGTrial.get_accuracy(trials=folded_trials)
