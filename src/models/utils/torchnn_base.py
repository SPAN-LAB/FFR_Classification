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
    
    def _core_train(self, *, 
        trials: list[EEGTrial], 
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float
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
            self.update_printed_training_status(print_content)
        
        return self.model
