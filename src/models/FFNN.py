from torch import nn
from math import floor

class Model(nn.Module):
    def __init__(self, input_size, num_classes = 4, dropout_p=0.5):
        super().__init__()
        hidden1 = floor(input_size / 2)  # ~400
        hidden2 = floor(hidden1 / 2)      # ~80
        hidden3 = floor(hidden2 / 2)      # ~40

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

            nn.Linear(hidden3, num_classes)
        )

    def forward(self, x):
        return self.model(x)