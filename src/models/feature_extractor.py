"""
SPAN Lab - FFR Classification

Filename: FeatureExtractor.py
Author(s): Cj
Description: Base classes for feature extraction and generation from EEG trials.
             FeatureExtractor  — transforms one trial at a time (stateless)
             FeatureGenerator  — learns from all trials first, then transforms (stateful)
             
             Current implementations:
                 FeatureGenerator — Autoencoder-based latent feature extraction
             
             Future implementations (drop in here):
                 PitchFeatureExtractor, SNRFeatureExtractor, etc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from numpy import typing as npt

from ..core import EEGSubject, EEGTrial


# ─────────────────────────────────────────────────────────────
# Base Classes
# ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Base class for stateless feature extraction.
    Transforms one trial at a time — no training needed.
    
    Subclass this for features like pitch, SNR, spectral correlation
    that can be computed per trial independently.
    """

    def _get_features(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """
        Transforms a single trial's data into features.
        data: an array of numbers
        returns: an array of numbers
        """
        raise NotImplementedError("This method needs to be implemented.")

    def transform(self, data: EEGTrial | list[EEGTrial] | EEGSubject):
        """
        Applies _get_features to each trial in-place.
        Accepts a single trial, list of trials, or a subject.
        """
        if isinstance(data, EEGTrial):
            data = [data]
        if isinstance(data, EEGSubject):
            data = data.trials
        for trial in data:
            trial.data = self._get_features(trial.data)


class FeatureGenerator(FeatureExtractor):
    """
    Base class for stateful feature generation.
    Learns from all trials first via _generate_features,
    then transforms each trial via _get_features.
    
    Currently implements Autoencoder-based feature extraction:
        - Trains a full Autoencoder on all provided trials (unsupervised, no labels)
        - Discards the decoder, keeps the frozen encoder
        - Replaces each trial's raw EEG (4,997 pts) with latent features (latent_dim pts)
    
    Subclass this for features that require learning from multiple trials,
    e.g. PCA, ICA, or other data-driven feature extraction methods.

    Usage:
        gen = FeatureGenerator(latent_dim=128, num_epochs=100)
        gen.generate_features(subject)     # accepts EEGSubject
        gen.generate_features(trials)      # accepts list[EEGTrial]
    """

    def __init__(self,
        latent_dim: int = 128,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ):
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.encoder = None  # set after _generate_features
        self.device = self._get_device()
    def transform(self, data: EEGTrial | list[EEGTrial] | EEGSubject):
        if isinstance(data, EEGTrial):
            data = [data]
        if isinstance(data, EEGSubject):
            data = data.trials
        dummy_timestamps = np.arange(self.latent_dim, dtype=np.float32)
        for trial in data:
            trial.data = self._get_features(trial.data)
            trial.timestamps = dummy_timestamps

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _generate_features(self, trial_data: list[npt.ArrayLike]):
        """
        Trains a full Autoencoder on all trial data (unsupervised).
        Stores the frozen encoder in self.encoder.
        Decoder is discarded after training.
        """
        X = torch.tensor(
            np.stack([np.array(d, dtype=np.float32).flatten() for d in trial_data]),
            dtype=torch.float32
        ).to(self.device)
        input_dim = X.shape[1]

        # Build encoder and decoder separately so encoder can be stored
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, self.latent_dim), nn.BatchNorm1d(self.latent_dim), nn.ReLU(),
        ).to(self.device)
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        ).to(self.device)

        # Link encoder and decoder for training — decoder is discarded after
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
                    f"FeatureGenerator | "
                    f"Epoch [{epoch + 1:3d}/{self.num_epochs}] | "
                    f"Loss: {epoch_loss / len(dataloader):.6f}"
                )

        # Freeze encoder — decoder goes out of scope and is garbage collected
        self.encoder.eval()

    def _get_features(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """
        Encodes a single trial's data using the stored frozen encoder.
        Returns a 1D array of shape (latent_dim,).
        """
        x = torch.tensor(
            np.array(data, dtype=np.float32).flatten(),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(x).cpu().numpy().flatten()

    def generate_features(self, trials: list[EEGTrial] | EEGSubject):
        """
        Step 1: Train autoencoder on all trials (_generate_features)
        Step 2: Replace each trial's data with latent features (transform)
        """
        if isinstance(trials, EEGSubject):
            trials = trials.trials
        trial_data = [trial.data for trial in trials]
        self._generate_features(trial_data)
        self.transform(trials)