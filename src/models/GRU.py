from torch import nn
import torch
from .utils import TorchNNBase


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        downsample: int = 2,   # 1=off, 2 is a good speed/accuracy tradeoff
    ):
        super().__init__()
        self.downsample = downsample

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,  # GRU dropout only between layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor):
        # Expect [B, T] or [B, T, 1]
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, T, 1]

        # Cheap speedup on long sequences
        if self.downsample > 1:
            x = x[:, :: self.downsample, :]  # [B, T/downsample, 1]

        # hidden_state: [num_layers*2, B, H] because bidirectional
        _, hidden_state = self.gru(x)

        # Take last layer forward/backward
        hidden_forward = hidden_state[-2]   # [B, H]
        hidden_backward = hidden_state[-1]  # [B, H]
        hidden = torch.cat([hidden_forward, hidden_backward], dim=1)  # [B, 2H]

        hidden = self.dropout(hidden)
        return self.fc(hidden)  # [B, num_classes]


class RNN_model(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)
        self.build()

    def build(self):
        input_size = int(self.training_options.get("input_size", 1))
        num_classes = int(self.training_options.get("n_classes", 4))

        # Moderate defaults: bidirectional but not huge
        hidden_size = int(self.training_options.get("hidden_size", 256))
        num_layers = int(self.training_options.get("num_layers", 1))
        dropout = float(self.training_options.get("dropout", 0.1))
        downsample = int(self.training_options.get("downsample", 2))

        self.model = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            downsample=downsample,
        ).to(self.device)

