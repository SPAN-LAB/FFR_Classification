"""
SPAN Lab - FFR Classification

Filename: torchnn_base.py
Author(s): TODO @Anu
Description: TODO @Anu
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ...core import EEGTrial
from ...core.utils.sampling import sds2
from .index_tracked_dataset import IndexTrackedDataset
from .model_interface import ModelInterface

from ...time import TimeKeeper
from ...printing.printing import Line
from random import shuffle
from numbers import Number


class TorchNNBase(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        # Initialize ``self.training_options``
        super().__init__(training_options)

        self.model: nn.Module | None = None
        self._best_weights = None
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
    
    def reset_seed(self):
        torch.manual_seed(0)
        torch.mps.manual_seed(0)
    
    def _store_best(self, best):
        self._best_weights = {}
        for k, v in best.state_dict().items():
            self._best_weights[k] = v.cpu().clone()

    def _restore_best(self):
        if self._best_weights is None:
            raise ValueError("self._best_weights = None unexpectedly")
        self.model.load_state_dict(self._best_weights)
        self.model.to(self.device)

    def build(self):
        raise NotImplementedError("This method needs to be implemented")

    def _core_avg_val_loss(self, *, 
        trials: list[EEGTrial], 
        batch_size: int
    ) -> float:
        """
        Determines the average loss for the input trials. 
        Called strictly inside the training loop in _core_train
        """
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        validation_loader = DataLoader(
            IndexTrackedDataset(trials=trials, inputs=self.required_inputs),
            batch_size=batch_size,
            shuffle=False
        )
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in validation_loader: 
                inputs = batch["x"].to(self.device)
                labels = batch["y"].to(self.device)

                logits = self.model(inputs)
                loss = criterion(logits, labels)
                
                # NOTE: 
                # loss.item() is the average loss per batch item
                # inputs.size(0) is the number of items in the batch
                # We do not use batch_size here, and instead use inputs.size(0), 
                # to protect against a last batch having fewer 
                # than <batch_size> items
                total_loss += loss.item() * inputs.size(0)
                
        self.model.train()
        
        average_loss = total_loss / len(trials)
        return average_loss

    def _core_infer(self, *, 
        trials: list[EEGTrial],
        batch_size: int
    ) -> list[dict[int, float]]:
        
        dataloader = DataLoader(
            IndexTrackedDataset(trials=trials, inputs=self.required_inputs),
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
        validation_trials: list[EEGTrial] | float | None = None,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        min_delta: float,
        patience: int
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
        
        must_validate = validation_trials is not None and validation_trials != 0
        if must_validate and isinstance(validation_trials, float):
            # Sample the validation trials from `trials` if a ratio is provided
            # and remove those from `trials`
            num_validation_trials = int(len(trials) * validation_trials)
            if num_validation_trials <= 0 or len(trials) - num_validation_trials <= 0:
                raise ValueError("The splitting of trials results in some group being empty")
            validation_trials = sds2(trials=trials, num_trials=num_validation_trials)
            for trial in validation_trials:
                trials.remove(trial)
        # Now, type of validation_trials is strictly either None or list[EEGTrial]
        
        train_loader = DataLoader(
            IndexTrackedDataset(trials=trials, inputs=self.required_inputs),
            batch_size=batch_size,
            shuffle=True
        )
        
        epoch_tk = TimeKeeper()
        epoch_tk.reset()
        epoch_tk.start()
        
        line = Line()
        
        self.model.train()
        self._reset_loss_trackers()
        for epoch_i in range(num_epochs):
            
            for batch in train_loader:
                
                inputs = batch["x"].to(self.device)
                labels = batch["y"].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            validation_loss = 0
            if must_validate:
                validation_loss = self._core_avg_val_loss(trials=validation_trials, batch_size=batch_size)
                self._record_loss(validation_loss, self.model)
                if not self._should_continue():
                    self._restore_best()
                    break
            
            epoch_tk.lap()
            # Formatting resets
            RESET     = "\033[0m"
            BOLD      = "\033[1m"
            UNDERLINE = "\033[4m"
            
            # Standard Colors
            
            # "Bright" Variants (Often look better in dark terminals)
            BR_GREEN   = "\033[92m"
            BR_YELLOW  = "\033[93m"
            BR_MAGENTA = "\033[95m"
            BR_CYAN    = "\033[96m"
            ORANGE = "\033[38;5;208m"
            msl_epoch = len(str(num_epochs)) # max string length 
            print_content = (
                f"Epoch {BR_CYAN}{epoch_i + 1:>{msl_epoch}}{RESET}/{BR_CYAN}{num_epochs}{RESET} | "
                f"Total {BR_GREEN}{epoch_tk.accumulated_duration:>7.3f}{RESET}s | "
                f"{BR_YELLOW}{epoch_tk.last_lap_duration:.3f}s{RESET}/epoch{RESET}"
            )
            if must_validate:
                print_content += f" | vloss={BR_MAGENTA}{validation_loss:.4f}{RESET} | low={BOLD}{ORANGE}{self._lowest_loss:.4f}{RESET}"
            
            line.place(print_content)
        
        return self.model
