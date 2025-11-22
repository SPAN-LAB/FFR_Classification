from torch import nn
import torch
from .utils import TorchNNBase


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor):
        # Expect [B, T] or [B, T, 1]
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, T, 1]

        # output: [B, T, 2H], hidden_state: [2, B, H] (for 1 layer, bidirectional)
        _, hidden_state = self.gru(x)

        hidden_forward = hidden_state[-2]  # [B, H]
        hidden_backward = hidden_state[-1]  # [B, H]
        hidden = torch.cat([hidden_forward, hidden_backward], dim=1)  # [B, 2H]

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # [B, num_classes]
        return logits


class RNN_model(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)
        self.build()

    def build(self):
        input_size = 1
        num_classes = 4
        hidden_size = 750  # or 128 if you want lighter GRU
        self.model = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
        ).to(self.device)
