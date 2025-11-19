from torch import nn
import torch
from .utils import TorchNNBase


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 128):
        super().__init__()
        # Input is [B, T] -> [B, 1, T]
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),  # T shrinks a lot here
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(0.0)  # keep 0 for now
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        feats = self.conv(x)  # [B, 64, T']
        feats = feats.transpose(1, 2)  # [B, T', 64]

        _, (h_n, _) = self.lstm(feats)  # h_n: [1, B, H]
        h = h_n[-1]  # [B, H]
        h = self.dropout(h)
        return self.fc(h)


class RNN_model(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)
        self.build()

    def build(self):
        num_classes = 4
        hidden_size = 128
        self.model = CNN_LSTM(num_classes=num_classes, hidden_size=hidden_size).to(
            self.device
        )
