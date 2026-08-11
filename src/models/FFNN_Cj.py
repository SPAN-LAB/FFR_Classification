import torch
from torch import nn
from .utils import TorchNNBase
from math import floor

class FFNN(TorchNNBase, nn.Module):
    def __init__(self, training_options: dict[str, any]):
        """
        NOTE: This method MUST be copied verbatim into every concrete PyTorch NN.
        """
        TorchNNBase.__init__(self, training_options)
        nn.Module.__init__(self)

    def build(self):
        # 1. Get Dimensions
        input_size = self.subject.trial_size
        output_size = self.subject.num_categories
        
        # 2. Hyperparameters (Extract or Default)
        # We use a wider initial layer to capture features, then taper down
        h1 = self.training_options.get("h1_size", floor(input_size * 0.75))
        h2 = self.training_options.get("h2_size", floor(h1 / 2))
        h3 = self.training_options.get("h3_size", floor(h2 / 2))
        
        dropout_rate = self.training_options.get("dropout", 0.5) # High dropout for EEG noise

        # 3. Model Architecture (Fine-Tuned)
        # Structure: Linear -> BatchNorm -> Activation -> Dropout
        self.model = nn.Sequential(
            # --- Layer 1 ---
            nn.Linear(input_size, h1),
            nn.BatchNorm1d(h1),       # Stabilizes learning
            nn.LeakyReLU(0.01),       # Better than ReLU for avoiding dead neurons
            nn.Dropout(dropout_rate), # Prevents overfitting
            
            # --- Layer 2 ---
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),

            # --- Layer 3 ---
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate / 2), # Lower dropout closer to output

            # --- Output Layer ---
            nn.Linear(h3, output_size)
        )
        
        # 4. Weight Initialization (He Initialization)
        # This helps the model converge much faster
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
        return self.model(x)