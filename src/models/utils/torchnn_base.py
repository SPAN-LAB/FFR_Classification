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
from .index_tracked_dataset import IndexTrackedDataset
from .model_interface import ModelInterface

from ...time import TimeKeeper


class TorchNNBase(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        # Initialize ``self.training_options``
        super().__init__(training_options)

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
    
    def reset_seed(self):
        torch.manual_seed(0)
        torch.mps.manual_seed(0)

    def build(self):
        raise NotImplementedError("This method needs to be implemented")

    def _core_infer(self, *, 
        trials: list[EEGTrial],
        batch_size: int
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
    
    def get_patience(self) -> int:
        if not isinstance(self.training_options, dict):
            return 5
        return self.training_options.get("patience", 5)

    def _core_train(self, *,
        trials: list[EEGTrial],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float
    ) -> nn.Module:

        # Split into 80% train, 20% val (keeping temporal order)
        split = int(len(trials) * 0.8)
        train_trials = trials[:split]
        val_trials = trials[split:]

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
            IndexTrackedDataset(trials=train_trials),
            batch_size=batch_size,
            shuffle=False
        )
        val_loader = DataLoader(
            IndexTrackedDataset(trials=val_trials),
            batch_size=batch_size,
            shuffle=False
        )

        epoch_tk = TimeKeeper()
        epoch_tk.reset()
        epoch_tk.start()

        best_val_loss = float("inf")
        best_weights = None
        epochs_no_improve = 0
        patience = self.get_patience()

        for epoch_i in range(num_epochs):

            # Training
            self.model.train()
            for batch in train_loader:
                inputs = batch["x"].to(self.device)
                labels = batch["y"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["x"].to(self.device)
                    labels = batch["y"].to(self.device)
                    logits = self.model(inputs)
                    val_loss += criterion(logits, labels).item()
            val_loss /= len(val_loader)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            epoch_tk.lap()
            print_content = (
                f"Epoch [ {epoch_i + 1}/{num_epochs} ]"
                f" | Val Loss: [ {val_loss:.4f} ]"
                f" | Best Val Loss: [ {best_val_loss:.4f} ]"
                f" | No improve: [ {epochs_no_improve}/{patience} ]"
                f" | [ {epoch_tk.last_lap_duration:.3f}s / epoch ]"
            )
            self.update_printed_training_status(print_content)

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch_i + 1}")
                break

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self.model
