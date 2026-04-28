"""
SPAN Lab - FFR Classification

Filename: Autoencoder.py
Author(s): Cj
Description: LOSO Autoencoder + SVM model, fully integrated into the AnalysisPipeline.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC

from ..core import EEGTrial
from .utils.torchnn_base import TorchNNBase


class _GlobalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _GlobalEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Autoencoder(TorchNNBase):
    """LOSO Autoencoder + SVM classifier."""

    needs_all_subjects: bool = True

    def __init__(self, training_options: dict):
        super().__init__(training_options)
        self.latent_dim: int = training_options.get("latent_dim", 128)
        self._encoder: _GlobalEncoder | None = None
        self._svm: SVC | None = None

    def build(self):
        pass

    def _pretrain_encoder(self, training_subjects: list) -> _GlobalEncoder:
        """Trains autoencoder on all LOSO training subjects. Returns frozen encoder."""
        all_data = []
        for subject in training_subjects:
            for trial in subject.trials:
                all_data.append(np.array(trial.data, dtype=np.float32).flatten())

        X = torch.tensor(np.array(all_data), dtype=torch.float32).to(self.device)
        input_dim = X.shape[1]

        autoencoder = _GlobalAutoencoder(input_dim, self.latent_dim).to(self.device)
        self.reset_seed()
        optimizer = optim.Adam(autoencoder.parameters(), lr=self.get_learning_rate())
        criterion = nn.MSELoss()
        dataloader = DataLoader(TensorDataset(X, X), batch_size=self.get_batch_size(), shuffle=True)
        num_epochs = self.get_num_epochs()

        autoencoder.train()
        for epoch in range(num_epochs):
            epoch_loss = sum(
                (lambda: (optimizer.zero_grad(), loss := criterion(autoencoder(bx), bx), loss.backward(), optimizer.step(), loss.item())[-1])()
                for bx, _ in dataloader
            )
            if (epoch + 1) % 50 == 0 or epoch == 0:
                self.update_printed_training_status(
                    f"AE Pretrain | Epoch [{epoch + 1:3d}/{num_epochs}] | Loss: {epoch_loss / len(dataloader):.6f}"
                )

        encoder = _GlobalEncoder(input_dim, self.latent_dim).to(self.device)
        encoder.encoder.load_state_dict(autoencoder.encoder.state_dict())
        encoder.eval()
        return encoder

    def _extract_features(self, trials: list) -> tuple:
        """Passes trials through the frozen encoder. Returns (features, labels)."""
        X = np.stack([np.array(t.data, dtype=np.float32).flatten() for t in trials])
        labels = np.array([int(getattr(t, "raw_label", getattr(t, "enumerated_label", 0))) for t in trials])
        with torch.no_grad():
            features = self._encoder(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return features, labels

    def _core_train(self, *, trials, validation_trials, num_epochs, batch_size, learning_rate, weight_decay, min_delta, patience):
        """Trains SVM on latent features. Stores result in self._svm."""
        X_train, y_train = self._extract_features(trials)
        self._svm = SVC(kernel="rbf", class_weight="balanced", probability=True)
        self._svm.fit(X_train, y_train)

    def _core_infer(self, *, trials, batch_size) -> list[dict[int, float]]:
        X_test, _ = self._extract_features(trials)
        probabilities = self._svm.predict_proba(X_test)
        labels_map = {
            int(float(k)): v 
            for k, v in self.subject.labels_map.items()
        }
        return [
            {
                labels_map[int(float(cls))]: float(p)
                for cls, p in zip(self._svm.classes_, row)
            }
            for _, row in enumerate(probabilities)
        ]

    def _core_avg_val_loss(self, *, trials, batch_size) -> float:
        return 0.0

    def evaluate(self, *, folded_trials: list = []) -> float:
        if self.subject is None:
            raise ValueError("No subject set.")
        if self.all_subjects is None:
            raise ValueError("No subjects list provided. Ensure needs_all_subjects=True.")

        training_subjects = [s for s in self.all_subjects if s.name != self.subject.name]
        print(f"\nAutoencoder | Pretraining on {len(training_subjects)} subjects (held out: {self.subject.name})")

        pretrain_start = time.time()
        self._encoder = self._pretrain_encoder(training_subjects)
        print(f"Autoencoder | Pretraining complete ({time.time() - pretrain_start:.1f}s)")

        if len(folded_trials) == 0:
            folded_trials = self.subject.folds

        folded_prediction_distributions = self._cross_validate(
            folded_trials=folded_trials,
            num_epochs=self.get_num_epochs(),
            batch_size=self.get_batch_size(),
            learning_rate=self.get_learning_rate(),
            weight_decay=self.get_weight_decay(),
            min_delta=self.get_min_delta(),
            patience=self.get_patience()
        )

        for i in range(len(folded_trials)):
            for j in range(len(folded_trials[i])):
                folded_trials[i][j].set_prediction_distribution(
                    enumerated_prediction_distribution=folded_prediction_distributions[i][j]
                )

        return EEGTrial.get_accuracy(trials=folded_trials)