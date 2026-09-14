"""
SPAN Lab - FFR Classification

Filename: autoencoder_latent.py
Author(s): Cj
Description: Autoencoder-based latent feature extractor for the feature registry.
             Subject-specific — trains on that subject's own trials (unsupervised).
             No label leakage since autoencoder never sees tone labels.
"""

import numpy as np
from numpy import typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class AutoencoderLatentExtractor:
    """
    Callable class that follows the feature registry signature:
        __call__(signal, fs) -> np.ndarray
    Has fit(signals) which must be called first to train the encoder.
    extract_features() detects fit() and calls it automatically.
    """

    def __init__(self,
        latent_dim: int = 128,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        self.latent_dim    = latent_dim
        self.num_epochs    = num_epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.encoder       = None
        self.device        = self._get_device()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, signals: list[npt.ArrayLike]):
        """
        Trains autoencoder on all signals, stores frozen encoder.
        Called once per subject by extract_features() before per-trial encoding.
        """
        X = torch.tensor(
            np.stack([np.array(s, dtype=np.float32).flatten() for s in signals]),
            dtype=torch.float32
        ).to(self.device)
        input_dim = X.shape[1]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, self.latent_dim), nn.BatchNorm1d(self.latent_dim), nn.ReLU(),
        ).to(self.device)
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        ).to(self.device)
        autoencoder = nn.Sequential(self.encoder, decoder)

        torch.manual_seed(42)
        optimizer = optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        dataloader = DataLoader(
            TensorDataset(X, X), batch_size=self.batch_size, shuffle=True
        )

        autoencoder.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                loss = criterion(autoencoder(batch_x), batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"autoencoder_latent | "
                    f"Epoch [{epoch+1:3d}/{self.num_epochs}] | "
                    f"Loss: {epoch_loss/len(dataloader):.6f}"
                )

        self.encoder.eval()
        print(f"autoencoder_latent | fit complete → {self.latent_dim}-dim features")

    def __call__(self, signal: npt.ArrayLike, fs: float) -> npt.ArrayLike:
        """
        Encodes a single trial using the trained encoder.
        Follows registry signature: (signal, fs) -> np.ndarray
        """
        if self.encoder is None:
            raise RuntimeError(
                "Must call fit() before encoding. "
                "Use extract_features(['autoencoder_latent']) in the pipeline."
            )
        x = torch.tensor(
            np.array(signal, dtype=np.float32).flatten(),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(x).cpu().numpy().flatten()


# Registry-ready singleton instance
autoencoder_latent = AutoencoderLatentExtractor(
    latent_dim=128,
    num_epochs=100,
    batch_size=64,
    learning_rate=0.001
)