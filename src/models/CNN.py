from .utils import TorchNNBase
from ..core.eeg_trial import EEGTrial

import torch
import torch.nn as nn


class _CNN1D(nn.Module):
    """
    Expects input [B, 1, T].
    """

    def __init__(self, n_classes: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x.unsqueeze(1)  # NOTE: changes shape to (N,1,T) as required by CNNs
        return self.net(x)


class CNNModel(TorchNNBase):
    def build(self) -> None:
        n_classes = int(self.hyperparameters.get("n_classes", 4))
        p_drop = float(self.hyperparameters.get("p_drop", 0.1))
        self.model = _CNN1D(n_classes=n_classes, p_drop=p_drop).to(self.device)

    def train(self, output_path: str) -> None:
        return None  # NOTE: Placeholder, to be implemented later

    def infer(self, trials: list[EEGTrial]):
        return None  # NOTE: Placeholder, to be implemented later
