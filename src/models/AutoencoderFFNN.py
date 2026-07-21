from .FFNN import FFNN
from math import floor
import torch.nn as nn

class AutoencoderFFNN(FFNN):
    required_inputs = ["autoencoder_latent"]

    def build(self):
        # Use latent_dim as input size instead of trial_size
        input_size  = len(self.subject.trials[0].features["autoencoder_latent"])
        output_size = self.subject.num_categories

        h1 = floor(input_size / 2)
        h2 = floor(h1 / 2)
        h3 = floor(h2 / 2)

        self.model = nn.Sequential(
            nn.Linear(input_size, h1), nn.ReLU(),
            nn.Linear(h1, h2),         nn.ReLU(),
            nn.Linear(h2, h3),         nn.ReLU(),
            nn.Linear(h3, output_size),
        )