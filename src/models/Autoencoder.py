"""
SPAN Lab - FFR Classification

Filename: Autoencoder.py
Author(s): Cj
Description: LOSO Autoencoder + SVM model, fully integrated into the AnalysisPipeline.

    This model combines two stages into a single evaluate() call:

    Stage 1 — LOSO Autoencoder Pretraining (in memory, no weights saved):
        A GlobalAutoencoder (Encoder + Decoder) is trained on all subjects
        except the current test subject. Training is unsupervised — the model
        learns to compress and reconstruct raw EEG signals into a 128-dim
        latent space.

    Stage 2 — SVM Classification:
        The Decoder is discarded. The frozen Encoder maps each trial of the
        test subject into a 128-dim latent vector. An SVM (RBF kernel) is
        then trained and evaluated using 5-fold cross-validation.

    Usage in demo.py (identical to all other models):

        p = (
            AnalysisPipeline()
            .load_subjects(SUBJECT_FILEPATHS)
            .trim_by_timestamp(start_time=0, end_time=float("inf"))
            .subaverage(5)
            .fold(5)
            .evaluate_model(
                model_name="Autoencoder",
                training_options={
                    "latent_dim": 128,
                    "num_epochs": 100,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                }
            )
        )
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from ..core import EEGSubject, EEGTrial
from .utils.model_interface import ModelInterface


# ─────────────────────────────────────────────────────────────
# Internal network definitions
# ─────────────────────────────────────────────────────────────

class _GlobalAutoencoder(nn.Module):
    """
    Full Autoencoder (Encoder + Decoder).
    Used only during pretraining — Decoder is discarded afterwards.
    """
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
    """
    Encoder-only module used at inference time.
    Weights are copied from _GlobalAutoencoder after pretraining.
    """
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


# ─────────────────────────────────────────────────────────────
# Autoencoder Model — conforms to ModelInterface
# ─────────────────────────────────────────────────────────────

class Autoencoder(ModelInterface):
    """
    LOSO Autoencoder + SVM classifier.

    Setting needs_all_subjects = True causes AnalysisPipeline.evaluate_model()
    to automatically call set_all_subjects() before evaluate(), giving this
    model access to every subject's data for LOSO pretraining.
    """

    needs_all_subjects: bool = True

    def __init__(self, training_options: dict):
        super().__init__(training_options)
        self.latent_dim: int = training_options.get("latent_dim", 128)

    # ── Helpers ───────────────────────────────────────────────

    def reset_seed(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ── Stage 1: LOSO Autoencoder pretraining ────────────────

    def _pretrain_encoder(
        self,
        training_subjects: list,
        device: torch.device
    ) -> _GlobalEncoder:
        """
        Trains a full Autoencoder on all training subjects (unsupervised).
        Copies the Encoder weights into a standalone _GlobalEncoder and
        returns it frozen. The Decoder is discarded.
        """

        # Gather all training trials into one flat numpy array
        all_data = []
        for subject in training_subjects:
            for trial in subject.trials:
                flat = np.array(trial.data, dtype=np.float32).flatten()
                all_data.append(flat)

        X = torch.tensor(np.array(all_data), dtype=torch.float32).to(device)
        input_dim = X.shape[1]

        # Build full autoencoder
        autoencoder = _GlobalAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim
        ).to(device)
        torch.manual_seed(42)
        np.random.seed(42)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            autoencoder.parameters(),
            lr=self.get_learning_rate()
        )

        dataset    = TensorDataset(X, X)
        dataloader = DataLoader(
            dataset,
            batch_size=self.get_batch_size(),
            shuffle=True
        )

        num_epochs = self.get_num_epochs()
        self.reset_seed()
        autoencoder.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstructed = autoencoder(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0 or epoch == 0:
                avg_loss = epoch_loss / len(dataloader)
                self.update_printed_training_status(
                    f"AE Pretrain | Epoch [{epoch + 1:3d}/{num_epochs}] "
                    f"| Loss: {avg_loss:.6f}"
                )

        # Transfer encoder weights → standalone encoder, discard decoder
        encoder = _GlobalEncoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim
        ).to(device)
        encoder.encoder.load_state_dict(autoencoder.encoder.state_dict())
        encoder.eval()  # Freeze

        return encoder

    # ── Stage 2: Feature extraction ───────────────────────────

    def _extract_features(
        self,
        trials: list,
        encoder: _GlobalEncoder,
        device: torch.device
    ):
        """
        Passes trials through the frozen encoder.

        Returns
        -------
        features : np.ndarray  (num_trials, latent_dim)
        labels   : np.ndarray  (num_trials,)
        """
        X = np.stack([
            np.array(t.data, dtype=np.float32).flatten()
            for t in trials
        ])
        labels = np.array([
            int(getattr(t, "raw_label", getattr(t, "enumerated_label", 0)))
            for t in trials
        ])

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            features = encoder(X_tensor).cpu().numpy()

        return features, labels

    # ── evaluate() — main entry point called by pipeline ──────

    def evaluate(self, *, folded_trials: list = []) -> float:
        """
        Full LOSO AE + SVM evaluation:
            1. Pretrain autoencoder on all subjects except self.subject (LOSO)
            2. Discard decoder, keep frozen encoder
            3. Run 5-fold SVM cross-validation on test subject using latent features
            4. Return overall accuracy
        """

        if self.subject is None:
            raise ValueError("No subject set. Call set_subject() before evaluate().")
        if self.all_subjects is None:
            raise ValueError(
                "No subjects list provided. "
                "Ensure needs_all_subjects=True and set_all_subjects() is called."
            )

        device = self._get_device()

        # ── Step 1: LOSO — exclude current test subject ───────
        training_subjects = [
            s for s in self.all_subjects
            if s.name != self.subject.name
        ]

        print(
            f"\nAutoencoder | Pretraining on {len(training_subjects)} subjects "
            f"(held out: {self.subject.name})"
        )

        pretrain_start = time.time()
        encoder = self._pretrain_encoder(training_subjects, device)
        print(
            f"Autoencoder | Pretraining complete "
            f"({time.time() - pretrain_start:.1f}s)"
        )

        # ── Step 2: Resolve folds ──────────────────────────────
        if len(folded_trials) == 0:
            folded_trials = self.subject.folds

        num_folds = len(folded_trials)

        # ── Step 3: 5-fold SVM cross-validation ───────────────
        total_correct = 0
        total_trials  = 0

        print(f"Autoencoder | Starting {num_folds}-fold SVM cross-validation...")

        for fold_i in range(num_folds):
            fold_start = time.time()

            test_trials  = folded_trials[fold_i]
            train_trials = [
                t
                for j, fold in enumerate(folded_trials)
                if j != fold_i
                for t in fold
            ]

            X_train, y_train = self._extract_features(train_trials, encoder, device)
            X_test,  y_test  = self._extract_features(test_trials,  encoder, device)

            svm = SVC(kernel="rbf", class_weight="balanced")
            svm.fit(X_train, y_train)
            predictions = svm.predict(X_test)

            fold_acc  = np.mean(predictions == y_test)
            fold_time = time.time() - fold_start
            print(
                f"Autoencoder | Fold [{fold_i + 1}/{num_folds}] "
                f"({fold_time:.1f}s) | Val Acc: {fold_acc:.3f}"
            )

            # Store predictions back onto trial objects
            for trial, pred in zip(test_trials, predictions):
                trial.prediction = int(pred)
                if int(getattr(trial, "raw_label", getattr(trial, "enumerated_label", 0))) == trial.prediction:
                    total_correct += 1
            total_trials += len(test_trials)

        final_accuracy = (total_correct / total_trials) * 100
        print(
            f"Autoencoder | Final Accuracy for {self.subject.name}: "
            f"{final_accuracy:.2f}%"
        )
        return final_accuracy

    # ── Stubs — required by ModelInterface, not used by Autoencoder ──

    def _core_train(self, *, trials, num_epochs, batch_size, learning_rate, weight_decay):
        pass

    def _core_infer(self, *, trials, batch_Size):
        pass

    def train(self, *, trials=[], pickle_to=None, overwrite=True):
        pass

    def infer(self, *, trials=[]):
        pass