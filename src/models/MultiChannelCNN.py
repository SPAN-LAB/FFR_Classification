"""

Multi-channel CNN that takes any combination of features
as separate channels. Each feature occupies one channel,
all padded to the same length (raw EEG length).
The CNN filter sees all channels simultaneously at each
timepoint — learns cross-feature patterns naturally.

"""

from .utils import TorchNNBase
import torch
import torch.nn as nn


class _MultiChannelCNN1D(nn.Module):
    """
    Expects input [B, C, T] where C = number of channels (features).
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=251, padding=125, bias=False),
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
        # x shape from IndexTrackedDataset:
        #   single input  → (B, T)     → unsqueeze to (B, 1, T)
        #   multi input   → (B, C, T)  → already correct
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class MultiChannelCNNModel(TorchNNBase):
    """
    Multi-channel CNN. Declare which features to use via required_inputs.
    All features are padded to raw EEG length by extract_features(concatenate=False).

    Examples:
        required_inputs = ["raw"]                    → 1 channel (same as CNNModel)
        required_inputs = ["raw", "autocorr"]        → 2 channels
        required_inputs = ["raw", "autocorr", "pitchtrack"] → 3 channels
    """

    required_inputs = [ "raw", "autocorr", "pitchtrack"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_channels = len(self.required_inputs)
        n_classes  = int(self.training_options.get("n_classes", 4))
        p_drop     = float(self.training_options.get("p_drop", 0.1))
        self.model = _MultiChannelCNN1D(
            n_channels=n_channels,
            n_classes=n_classes,
            p_drop=p_drop
        ).to(self.device)