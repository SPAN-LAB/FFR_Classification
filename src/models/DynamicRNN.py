"""
SPAN Lab - FFR Classification

Filename: DynamicRNN.py
Author(s): Cj
Description: Dynamic multivariate LSTM that stacks multiple features as
             channels per timestep — the natural way to use RNNs for
             multi-feature time series.

             Unlike the original RNN (input_size=1, one timestep at a time),
             this reads multiple features simultaneously at each timestep:
                 (B, T, n_channels) where each channel is one feature.

             Feature alignment strategy:
                 - Temporal features (raw, autocorr) at full length T
                 - Shorter temporal features (pitchtrack) → interpolated to T
                 - Non-temporal features (autoencoder_latent) → broadcast to T
                   (same 128 values repeated at every timestep)

             Change required_inputs to use different features.
             n_channels = len(required_inputs) automatically.

Architecture:
    All features aligned to length T → stacked (B, T, n_channels)
    → Bidirectional LSTM(input_size=n_channels, hidden=256)
    → concat(forward, backward) → Linear → 4 classes

Usage:
    .extract_features(["pitchtrack", "autocorr", "autoencoder_latent"])
    .evaluate_model("DynamicRNN", training_options={
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "patience": 20,
        "min_delta": 0.001,
        "hidden_size": 256,
        "p_drop": 0.2,
    })
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .utils.torchnn_base import TorchNNBase
from ..core.eeg_trial import EEGTrial
from ..core.utils.sampling import sds2

# Features shorter than raw but still temporal — interpolate to raw length
TEMPORAL_FEATURES    = {"raw", "pitchtrack", "autocorr"}
# Features that are not temporal — broadcast to all timesteps
BROADCAST_FEATURES   = {"autoencoder_latent"}


class _DynamicLSTMNet(nn.Module):
    """
    Bidirectional LSTM that reads a multivariate sequence.
    Input: (B, T, n_channels) — T timesteps, n_channels features per step.
    """
    def __init__(self,
        n_channels: int,
        hidden_size: int,
        n_classes: int,
        p_drop: float,
        num_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_drop if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p_drop)
        self.fc      = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_channels)
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers * 2, B, hidden_size)
        # Take last layer forward and backward
        h_forward  = hidden[-2]   # (B, hidden_size)
        h_backward = hidden[-1]   # (B, hidden_size)
        h = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden_size*2)
        h = self.dropout(h)
        return self.fc(h)


class DynamicRNNModel(TorchNNBase):
    """
    Dynamic multivariate RNN.
    Change required_inputs to use different features.
    All features are aligned to the same sequence length T and stacked
    as channels: (B, T, n_channels).
    """

    # ── Change this to use different features ──────────────────
    required_inputs = ["raw", "pitchtrack", "autocorr", "autoencoder_latent"]
    # ───────────────────────────────────────────────────────────

    def __init__(self, training_options: dict):
        TorchNNBase.__init__(self, training_options)
        self._net = None

    def build(self):
        pass  # built lazily

    def _get_sequence_length(self, trials: list[EEGTrial]) -> int:
        """Use raw EEG length as the reference sequence length."""
        return len(np.array(trials[0].data).flatten())

    def _align_feature(self,
        arr: np.ndarray,
        target_len: int,
        name: str
    ) -> np.ndarray:
        """
        Align a feature array to target_len.
        - Same length: return as-is
        - Temporal but shorter (pitch): interpolate
        - Non-temporal (AE latent): broadcast to every timestep
        """
        arr = arr.flatten().astype(np.float32)
        if len(arr) == target_len:
            return arr

        if name in BROADCAST_FEATURES:
            # Repeat the feature vector at every timestep
            # Shape: (target_len, len(arr)) — but we want (target_len,) for stacking
            # Instead tile the values: repeat arr enough times to fill target_len
            # This gives the AE summary as a repeated signal
            repeats = target_len // len(arr) + 1
            tiled = np.tile(arr, repeats)[:target_len]
            return tiled.astype(np.float32)

        if name in TEMPORAL_FEATURES:
            # Interpolate shorter temporal feature to target_len
            tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            interp = F.interpolate(tensor, size=target_len, mode="linear",
                                   align_corners=False)
            return interp.squeeze().numpy().astype(np.float32)

        # Default: interpolate
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
        interp = F.interpolate(tensor, size=target_len, mode="linear",
                               align_corners=False)
        return interp.squeeze().numpy().astype(np.float32)

    def _get_sequence_tensors(self,
        trials: list[EEGTrial]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build multivariate sequence tensor (B, T, n_channels) from all features.
        """
        T = self._get_sequence_length(trials)
        B = len(trials)
        C = len(self.required_inputs)

        sequences = np.zeros((B, T, C), dtype=np.float32)

        for b, trial in enumerate(trials):
            for c, name in enumerate(self.required_inputs):
                if name == "raw":
                    arr = np.array(trial.data, dtype=np.float32).flatten()
                else:
                    arr = np.array(trial.features[name], dtype=np.float32).flatten()
                sequences[b, :, c] = self._align_feature(arr, T, name)

        x      = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        labels = torch.tensor(
            [t.enumerated_label for t in trials], dtype=torch.long
        ).to(self.device)
        return x, labels

    def _build_net(self, trials: list[EEGTrial]):
        n_channels  = len(self.required_inputs)
        hidden_size = int(self.training_options.get("hidden_size", 256))
        n_classes   = self.subject.num_categories
        p_drop      = float(self.training_options.get("p_drop", 0.2))
        num_layers  = int(self.training_options.get("num_layers", 2))

        print(f"DynamicRNN | n_channels={n_channels} features={self.required_inputs}")
        print(f"DynamicRNN | hidden={hidden_size} layers={num_layers} bidir=True")

        self._net = _DynamicLSTMNet(
            n_channels=n_channels,
            hidden_size=hidden_size,
            n_classes=n_classes,
            p_drop=p_drop,
            num_layers=num_layers,
        ).to(self.device)
        self.model = self._net

    def _core_train(self, *,
        trials, validation_trials, num_epochs,
        batch_size, learning_rate, weight_decay, min_delta, patience
    ):
        self._build_net(trials)
        optimizer = optim.AdamW(
            self._net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        must_validate = validation_trials is not None and validation_trials != 0
        if must_validate and isinstance(validation_trials, float):
            n_val = int(len(trials) * validation_trials)
            if n_val <= 0 or len(trials) - n_val <= 0:
                must_validate = False
            else:
                validation_trials = sds2(trials=trials, num_trials=n_val)
                for t in validation_trials:
                    trials.remove(t)

        train_x, train_y = self._get_sequence_tensors(trials)
        if must_validate:
            val_x, val_y = self._get_sequence_tensors(validation_trials)

        n = len(trials)
        self._reset_loss_trackers()
        self._net.train()

        for epoch in range(num_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = self._net(train_x[idx])
                loss   = criterion(logits, train_y[idx])
                loss.backward()
                optimizer.step()

            if must_validate:
                self._net.eval()
                with torch.no_grad():
                    val_loss = criterion(self._net(val_x), val_y).item()
                self._net.train()
                self._record_loss(val_loss, self._net)
                if not self._should_continue():
                    self._restore_best()
                    break

    def _core_infer(self, *, trials, batch_size):
        x, _ = self._get_sequence_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            probs = torch.softmax(self._net(x), dim=1).cpu().numpy()
        return [{i: float(p) for i, p in enumerate(row)} for row in probs]

    def _core_avg_val_loss(self, *, trials, batch_size) -> float:
        x, y = self._get_sequence_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            loss = nn.CrossEntropyLoss()(self._net(x), y).item()
        self._net.train()
        return loss