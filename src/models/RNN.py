from torch import nn
import torch
from .utils import TorchNNBase


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        _, (hidden_state, _) = self.lstm(x)
        hidden_forward = hidden_state[-2]
        hidden_backward = hidden_state[-1]
        hidden = torch.cat([hidden_forward, hidden_backward], dim=1)

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


class RNN_model(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self):
        input_size = 1
        num_classes = 4
        hidden_size = 750
        self.model = RNN(
            input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
        ).to(self.device)
