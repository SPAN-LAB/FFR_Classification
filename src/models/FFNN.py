from torch import nn
from math import floor
from .model_interface import TorchNNBase
from ..core.eeg_trial import EEGTrial
from typing import Any


class _FFNN(nn.Module):
    def __init__(self, input_size, num_classes=4, dropout_p=0.5):
        super().__init__()
        hidden1 = floor(input_size / 2)  # ~400
        hidden2 = floor(hidden1 / 2)  # ~80
        hidden3 = floor(hidden2 / 2)  # ~40

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(hidden3, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class FFNNModel(TorchNNBase):
    def build(self):
        n_classes = int(self.hyperparameters.get("n_classes", 4))
        if self.subject is not None:
            input_size = int(len(self.subject.trials[0].timestamps))
        else:
            raise RuntimeError(
                "Subject has not been set, set subject before running build."
            )
        self.model = _FFNN(input_size=input_size, num_classes=n_classes)

    def train(self, output_path: str) -> None:
        return None  # NOTE: Placeholder, to be implemented later

    def infer(self, trials: list[EEGTrial]):
        return None  # NOTE: Placeholder, to be implemented later
