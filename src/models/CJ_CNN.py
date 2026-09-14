from .utils import TorchNNBase
from ..core.eeg_trial import EEGTrial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class _CNN1D(nn.Module):
    """
    Expects input [B, 1, T].
    Optimized for raw FFR end-to-end learning.
    """
    def __init__(self, n_classes: int = 4, p_drop: float = 0.25): 
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 50ms window (251 timepoints at 5000Hz)
            nn.Conv1d(1, 64, kernel_size=251, padding=125, bias=False), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2), 

            nn.Conv1d(64, 128, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2), 

            nn.Conv1d(128, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2), 

            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 'x' right here is purely the numbers from ffr_nodss!
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class CNNModel(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", self.subject.num_categories if self.subject else 4))
        p_drop = float(self.training_options.get("p_drop", 0.1))
        self.model = _CNN1D(n_classes=n_classes, p_drop=p_drop).to(self.device)

    def setup_dynamic_loss(self, train_trials: list[EEGTrial]) -> None:
        """Calculates dynamic class weights based on the actual dataset imbalance."""
        n_classes = int(self.training_options.get("n_classes", self.subject.num_categories if self.subject else 4))
        
        # Extract just the labels for grading
        labels = [trial.mapped_label for trial in train_trials]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        class_counts = torch.bincount(labels_tensor, minlength=n_classes)
        class_counts = torch.where(class_counts == 0, torch.tensor(1), class_counts)
        
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
        
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))

    def train(self, output_path: str = "") -> None:
        # 1. Setup the loss function with our Dynamic Weights
        self.setup_dynamic_loss(self.train_trials)
        
        # 2. Extract the Data (ffr_nodss) and Labels into PyTorch Tensors
        # NOTE: Check if your trial object uses .data or .signal for the numbers
        x_data = torch.stack([torch.tensor(trial.data, dtype=torch.float32) for trial in self.train_trials])
        y_labels = torch.tensor([trial.mapped_label for trial in self.train_trials], dtype=torch.long)
        
        # 3. Create a DataLoader to feed the network in batches
        batch_size = self.training_options.get("batch_size", 32)
        dataset = TensorDataset(x_data, y_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 4. Setup the Optimizer
        lr = self.training_options.get("learning_rate", 0.0005)
        wd = self.training_options.get("weight_decay", 0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        
        # 5. The Training Loop
        epochs = self.training_options.get("num_epochs", 50)
        self.model.train()
        
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # The model ONLY sees the ffr_nodss numbers here
                predictions = self.model(batch_x) 
                
                # The Loss Function compares the predictions to the true labels here
                loss = self.criterion(predictions, batch_y) 
                
                loss.backward()
                optimizer.step()

    def infer(self, trials: list[EEGTrial]):
        self.model.eval()
        x_data = torch.stack([torch.tensor(trial.data, dtype=torch.float32) for trial in trials]).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x_data)
            
        return torch.argmax(predictions, dim=1).cpu().numpy()