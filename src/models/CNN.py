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
        if x.ndim == 2:
            x = x.unsqueeze(1)  # NOTE: changes shape to (N,1,T) as required by CNNs
        return self.net(x)


class CNNModel(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", 4))
        p_drop = float(self.training_options.get("p_drop", 0.1))
        self.model = _CNN1D(n_classes=n_classes, p_drop=p_drop).to(self.device)

    # def train(self, output_path: str) -> None:
    #     return None  # NOTE: Placeholder, to be implemented later

    # def infer(self, trials: list[EEGTrial]):
    #     return None  # NOTE: Placeholder, to be implemented later