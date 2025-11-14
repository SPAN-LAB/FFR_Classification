from torch import nn
from .utils import TorchNNBase

from math import floor

class FFNN(TorchNNBase, nn.Module):
    def __init__(self, training_options: dict[str, any]):
        """
        NOTE: This method MUST be copied verbatim into every concrete PyTorch NN.
        """
        TorchNNBase.__init__(self,training_options)
        nn.Module.__init__(self)

    def build(self):
        # TODO hardcoded values; remove later
        input_size = len(self.subject.trials[0].data)
        num_classes = 4 
        """
        This part should become:

            input_size = subject.num_datapoints
            output_size = subject.num_possible_labels

        """
        ########################################

        dropout_p = 0.5

        # Calculate hidden layer sizes based on the input size so they gradually decrease
        hidden1 = floor(input_size / 2)  # ~400
        hidden2 = floor(hidden1 / 2)  # ~80
        hidden3 = floor(hidden2 / 2)  # ~40

        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden3, num_classes),
        )

    def forward(self, x):
        return self.model(x)