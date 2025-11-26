from torch import nn
from .utils import TorchNNBase

from math import floor

class FFNN(TorchNNBase, nn.Module):
    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self,training_options)
        nn.Module.__init__(self)

    def build(self):
        input_size = self.subject.trial_size
        output_size = self.subject.num_categories
        dropout_probability = 0.0
        h1 = floor(input_size / 2)
        h2 = floor(h1 / 2)
        h3 = floor(h2 / 2)

        self.model = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(h3, output_size),
        )

    def forward(self, x):
        return self.model(x)