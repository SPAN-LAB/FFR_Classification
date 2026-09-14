"""
SPAN Lab - FFR Classification

Filename: DynamicCRNN.py
Author(s): Cj
Description: Dynamic CNN-RNN hybrid. A lightweight CNN front-end processes each
             temporal feature independently, then all CNN outputs are stacked as
             channels and fed into a bidirectional LSTM.

             Non-temporal features (autoencoder_latent) skip the CNN and are
             broadcast to match the LSTM sequence length.

             Change required_inputs to use different features — CNN branches
             and sequence alignment happen automatically.

Architecture:
    raw (3277)     → LightCNN → (B, 64, ~204) ─┐
    autocorr(3277) → LightCNN → (B, 64, ~204) ─┤ stack → (B, ~204, n_channels*64)
    pitch (265)    → LightCNN → (B, 32, ~204) ─┤ (all aligned to same T)
    AE (128)       → broadcast → (B, 128, ~204)─┘
    → BiLSTM(input_size=total_channels) → concat → Linear → 4 classes

Usage:
    .extract_features(["pitchtrack", "autocorr", "autoencoder_latent"])
    .evaluate_model("DynamicCRNN", training_options={
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "patience": 20,
        "min_delta": 0.001,
        "hidden_size": 256,
        "num_layers": 2,
        "p_drop": 0.2,
    })
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils.torchnn_base import TorchNNBase
from ..core.eeg_trial import EEGTrial
from ..core.utils.sampling import sds2

TEMPORAL_FEATURES  = {"raw", "pitchtrack", "autocorr", "spectrogram"}
BROADCAST_FEATURES = {"autoencoder_latent"}


class _TemporalCNNEncoder(nn.Module):
    """
    Lightweight CNN that compresses a 1D temporal feature.
    Long features (>500): 2 conv+pool layers → T/16
    Short features (200-500): 1 conv+pool layer → T/4
    Output: (B, out_channels, T_compressed)
    """

    def __init__(self, input_len: int):
        super().__init__()
        if input_len > 500:
            self.out_channels = 64
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=51, padding=25, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(4),
                nn.Conv1d(32, 64, kernel_size=25, padding=12, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(), nn.AvgPool1d(4),
            )
            self.compress_ratio = 16
        else:
            self.out_channels = 32
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=25, padding=12, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(4),
            )
            self.compress_ratio = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = x.unsqueeze(1)   # (B, 1, T)
        return self.net(x)   # (B, out_channels, T_compressed)


class _DynamicCRNNNet(nn.Module):
    """
    Dynamic CRNN network.
    Each temporal feature gets its own CNN encoder.
    Non-temporal features are broadcast to match sequence length.
    All channels concatenated → BiLSTM → classifier.
    """

    def __init__(self,
        feature_names: list[str],
        feature_sizes: dict[str, int],
        hidden_size: int,
        n_classes: int,
        p_drop: float,
        num_layers: int,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.feature_sizes = feature_sizes

        # Build CNN encoders for temporal features
        self.cnn_encoders = nn.ModuleDict()
        total_channels = 0

        for name in feature_names:
            size = feature_sizes[name]
            if name in TEMPORAL_FEATURES:
                encoder = _TemporalCNNEncoder(size)
                self.cnn_encoders[name] = encoder
                total_channels += encoder.out_channels
            else:
                # Non-temporal: broadcast raw values as channels
                total_channels += size

        print(f"DynamicCRNN | total LSTM input channels: {total_channels}")

        self.lstm = nn.LSTM(
            input_size=total_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_drop if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p_drop)
        self.fc      = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        B = next(iter(inputs.values())).shape[0]
        encoded = []
        target_len = None

        # Process temporal features through CNN first
        for name in self.feature_names:
            if name in TEMPORAL_FEATURES and name in self.cnn_encoders:
                feat = inputs[name]           # (B, T)
                out  = self.cnn_encoders[name](feat)  # (B, C, T_compressed)
                encoded.append(("temporal", name, out))
                if target_len is None:
                    target_len = out.shape[2]

        # Align all temporal outputs to same length
        aligned = []
        for kind, name, out in encoded:
            if out.shape[2] != target_len:
                out = F.interpolate(out, size=target_len, mode="linear",
                                    align_corners=False)
            aligned.append(out)  # (B, C, T)

        # Broadcast non-temporal features to target_len
        for name in self.feature_names:
            if name in BROADCAST_FEATURES:
                feat = inputs[name]           # (B, D)
                # Tile D values across target_len timesteps
                feat = feat.unsqueeze(2).expand(B, feat.shape[1], target_len)
                aligned.append(feat)          # (B, D, T)

        # Stack all channels: (B, total_channels, T)
        x = torch.cat(aligned, dim=1)

        # Transpose for LSTM: (B, T, total_channels)
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)
        h_forward  = hidden[-2]
        h_backward = hidden[-1]
        h = torch.cat([h_forward, h_backward], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class DynamicCRNNModel(TorchNNBase):
    """
    Dynamic CNN-RNN hybrid model.
    Change required_inputs to use different features.
    CNN branches created automatically for temporal features.
    Non-temporal features broadcast across timesteps.
    """

    # ── Change this to use different features ──────────────────
    required_inputs = ["raw", "pitchtrack", "autocorr", "autoencoder_latent"]
    # ───────────────────────────────────────────────────────────

    def __init__(self, training_options: dict):
        TorchNNBase.__init__(self, training_options)
        self._net = None

    def build(self):
        pass  # built lazily

    def _get_feature_sizes(self, trials: list[EEGTrial]) -> dict[str, int]:
        sizes = {}
        trial = trials[0]
        for name in self.required_inputs:
            if name == "raw":
                sizes[name] = len(np.array(trial.data).flatten())
            else:
                sizes[name] = len(np.array(trial.features[name]).flatten())
        return sizes

    def _build_net(self, trials: list[EEGTrial]):
        feature_sizes = self._get_feature_sizes(trials)
        n_classes     = self.subject.num_categories
        hidden_size   = int(self.training_options.get("hidden_size", 256))
        p_drop        = float(self.training_options.get("p_drop", 0.2))
        num_layers    = int(self.training_options.get("num_layers", 2))

        print(f"DynamicCRNN | features: {self.required_inputs}")
        for name, size in feature_sizes.items():
            kind = "temporal" if name in TEMPORAL_FEATURES else "broadcast"
            print(f"  {name}: {size} pts ({kind})")

        self._net = _DynamicCRNNNet(
            feature_names=self.required_inputs,
            feature_sizes=feature_sizes,
            hidden_size=hidden_size,
            n_classes=n_classes,
            p_drop=p_drop,
            num_layers=num_layers,
        ).to(self.device)
        self.model = self._net

    def _get_feature_tensors(self,
        trials: list[EEGTrial]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        feats = {}
        for name in self.required_inputs:
            if name == "raw":
                arrays = [np.array(t.data, dtype=np.float32).flatten() for t in trials]
            else:
                arrays = [np.array(t.features[name], dtype=np.float32).flatten() for t in trials]
            feats[name] = torch.tensor(
                np.stack(arrays), dtype=torch.float32
            ).to(self.device)

        labels = torch.tensor(
            [t.enumerated_label for t in trials], dtype=torch.long
        ).to(self.device)
        return feats, labels

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

        train_feats, train_labels = self._get_feature_tensors(trials)
        if must_validate:
            val_feats, val_labels = self._get_feature_tensors(validation_trials)

        n = len(trials)
        self._reset_loss_trackers()
        self._net.train()

        for epoch in range(num_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx          = indices[start:start + batch_size]
                batch_feats  = {name: train_feats[name][idx] for name in self.required_inputs}
                batch_labels = train_labels[idx]

                optimizer.zero_grad(set_to_none=True)
                logits = self._net(batch_feats)
                loss   = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            if must_validate:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(val_feats)
                    val_loss   = criterion(val_logits, val_labels).item()
                self._net.train()
                self._record_loss(val_loss, self._net)
                if not self._should_continue():
                    self._restore_best()
                    break

    def _core_infer(self, *, trials, batch_size):
        feats, _ = self._get_feature_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            probs = torch.softmax(self._net(feats), dim=1).cpu().numpy()
        return [{i: float(p) for i, p in enumerate(row)} for row in probs]

    def _core_avg_val_loss(self, *, trials, batch_size) -> float:
        feats, labels = self._get_feature_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            loss = nn.CrossEntropyLoss()(self._net(feats), labels).item()
        self._net.train()
        return loss