"""
SPAN Lab - FFR Classification

Filename: DynamicCNN.py
Author(s): Cj
Description: Dynamic multi-branch CNN that automatically creates one branch
             per feature based on required_inputs.

             Temporal features (raw, autocorr, pitchtrack) get Conv1d branches
             sized appropriately for their length.
             Non-temporal features (autoencoder_latent) get a Linear branch.

             No padding — each feature processed at its natural size.
             All branches output embed_dim → concat → classifier.

Architecture:
    raw (3277)             → DeepConvBranch   → 64
    autocorr (3277)        → DeepConvBranch   → 64
    pitchtrack (265)       → LightConvBranch  → 64
    autoencoder_latent(128)→ LinearBranch     → 64
                             concat(256) → Linear(256→4)

Usage:
    .extract_features(["pitchtrack", "autocorr", "autoencoder_latent"])
    .evaluate_model("DynamicCNN", training_options={
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "patience": 20,
        "min_delta": 0.001,
        "embed_dim": 64,
    })

To change features — just update required_inputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.torchnn_base import TorchNNBase
from ..core.eeg_trial import EEGTrial
from ..core.utils.sampling import sds2

# Threshold below which we use a lighter CNN (e.g. pitch at 265)
LIGHT_CNN_THRESHOLD = 500
# Threshold below which we skip CNN entirely and use Linear (e.g. AE at 128)
LINEAR_THRESHOLD    = 200


class _DeepConvBranch(nn.Module):
    """
    Full CNN branch for long temporal features (raw, autocorr ~3277 pts).
    Same architecture as the existing CNNModel.
    """
    def __init__(self, embed_dim: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=251, padding=125, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AvgPool1d(2),

            nn.Conv1d(64, 128, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AvgPool1d(2),

            nn.Conv1d(128, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AvgPool1d(2),

            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)   # (B, T) → (B, 1, T)
        return self.net(x)


class _LightConvBranch(nn.Module):
    """
    Lighter CNN branch for shorter temporal features (pitchtrack ~265 pts).
    Smaller kernels suited to shorter sequences.
    """
    def __init__(self, embed_dim: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, padding=25, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(), nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class _LinearBranch(nn.Module):
    """
    Linear branch for non-temporal features (autoencoder_latent ~128 pts).
    No convolution — just compress to embed_dim.
    """
    def __init__(self, input_size: int, embed_dim: int, p_drop: float):
        super().__init__()
        hidden = min(256, max(embed_dim, input_size // 2))
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, embed_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DynamicMultiBranchCNNNet(nn.Module):
    """
    Dynamic multi-branch CNN.
    Branch type selected automatically based on feature size:
        > 500 pts  → DeepConvBranch  (raw, autocorr)
        200-500 pts→ LightConvBranch (pitchtrack)
        < 200 pts  → LinearBranch   (autoencoder_latent)
    """

    def __init__(self,
        feature_names: list[str],
        input_sizes: dict[str, int],
        embed_dim: int,
        n_classes: int,
        p_drop: float,
    ):
        super().__init__()
        self.feature_names = feature_names

        branches = {}
        for name, size in input_sizes.items():
            if size < LINEAR_THRESHOLD:
                branches[name] = _LinearBranch(size, embed_dim, p_drop)
                branch_type = "Linear"
            elif size < LIGHT_CNN_THRESHOLD:
                branches[name] = _LightConvBranch(embed_dim, p_drop)
                branch_type = "LightCNN"
            else:
                branches[name] = _DeepConvBranch(embed_dim, p_drop)
                branch_type = "DeepCNN"
            print(f"  DynamicCNN | {name} ({size} pts) → {branch_type} branch → {embed_dim}")

        self.branches = nn.ModuleDict(branches)

        total = embed_dim * len(feature_names)
        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(total, total // 2), nn.ReLU(),
            nn.Linear(total // 2, n_classes),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = [
            self.branches[name](inputs[name])
            for name in self.feature_names
        ]
        return self.classifier(torch.cat(embeddings, dim=1))


class DynamicCNNModel(TorchNNBase):
    """
    Dynamic multi-branch CNN.
    Change required_inputs to use different features.
    Branch type is chosen automatically based on feature size.
    """

    # ── Change this to use different features ──────────────────
    required_inputs = ["raw", "pitchtrack", "autocorr", "autoencoder_latent"]
    # ───────────────────────────────────────────────────────────

    def __init__(self, training_options: dict):
        TorchNNBase.__init__(self, training_options)
        self.embed_dim = int(training_options.get("embed_dim", 64))
        self._net = None

    def build(self):
        pass  # built lazily

    def _get_input_sizes(self, trials: list[EEGTrial]) -> dict[str, int]:
        sizes = {}
        trial = trials[0]
        for name in self.required_inputs:
            if name == "raw":
                sizes[name] = len(np.array(trial.data).flatten())
            else:
                sizes[name] = len(np.array(trial.features[name]).flatten())
        return sizes

    def _build_net(self, trials: list[EEGTrial]):
        input_sizes = self._get_input_sizes(trials)
        n_classes   = self.subject.num_categories
        p_drop      = float(self.training_options.get("p_drop", 0.1))

        self._net = _DynamicMultiBranchCNNNet(
            feature_names=self.required_inputs,
            input_sizes=input_sizes,
            embed_dim=self.embed_dim,
            n_classes=n_classes,
            p_drop=p_drop,
        ).to(self.device)
        self.model = self._net

    def _get_feature_tensors(self, trials: list[EEGTrial]) -> tuple[dict, torch.Tensor]:
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
            logits = self._net(feats)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return [{i: float(p) for i, p in enumerate(row)} for row in probs]

    def _core_avg_val_loss(self, *, trials, batch_size) -> float:
        feats, labels = self._get_feature_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(feats)
            loss   = nn.CrossEntropyLoss()(logits, labels).item()
        self._net.train()
        return loss