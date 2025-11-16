from torch import nn
import torch
from .utils import TorchNNBase


class RNN(TorchNNBase, nn.Module):
    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        nn.Module.__init__(self)

    def build(self):
        input_size = self.subject.trial_size
        num_classes = self.subject.num_categories
        hidden_size = 750

        self.model = nn.LSTM(
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

        output, (hidden_state, cell_state) = self.model(x)
        hidden_forward = hidden_state[-2]
        hidden_backward = hidden_state[-1]
        hidden = torch.cat([hidden_forward, hidden_backward], dim=1)

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits
