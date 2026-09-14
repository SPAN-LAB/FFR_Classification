from .utils import TorchNNBase

import torch
import torch.nn as nn


class _PitchCNN1D(nn.Module):
    """
    Same architecture as CNN but with smaller kernels suited to the pitch track
    input (~265 samples vs ~4997 for raw).

    Expects input [B, 1, T].
    """

    def __init__(self, n_classes: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=51, padding=25, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, T) → (B, 1, T)
        return self.net(x)


class PitchCNNModel(TorchNNBase):
    required_inputs = ["pitchtrack"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", 4))
        p_drop    = float(self.training_options.get("p_drop", 0.1))
        self.model = _PitchCNN1D(n_classes=n_classes, p_drop=p_drop).to(self.device)
