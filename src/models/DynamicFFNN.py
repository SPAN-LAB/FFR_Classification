"""
SPAN Lab - FFR Classification

Filename: DynamicMultiBranchFFNN.py
Author(s): Cj
Description: A dynamic multi-branch FFNN that automatically creates one branch
             per feature based on required_inputs. Each feature gets its own
             dedicated Linear layers at its natural size — no padding needed.

             To use different features, just change required_inputs.
             Branches are created automatically at the right size.

Architecture:
    For each feature in required_inputs:
        feature (natural size) → Linear(size→256) → ReLU → Linear(256→64) → ReLU
    All branch outputs → concat → Linear(64*n→32) → ReLU → Linear(32→n_classes)

Usage in demo.py:
    .extract_features(["pitchtrack", "autocorr", "autoencoder_latent"])
    .evaluate_model("DynamicMultiBranchFFNN", training_options={
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "patience": 50,
        "min_delta": 0.001,
        "embed_dim": 64,   # output size of each branch
    })

To change features — just update required_inputs:
    required_inputs = ["raw"]
    required_inputs = ["raw", "pitchtrack"]
    required_inputs = ["autoencoder_latent", "pitchtrack"]
    required_inputs = ["raw", "pitchtrack", "autocorr", "autoencoder_latent"]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.torchnn_base import TorchNNBase
from ..core.eeg_trial import EEGTrial
from ..core.utils.sampling import sds2


class _Branch(nn.Module):
    """
    Single FFNN branch for one feature.
    Compresses input_size → embed_dim through two linear layers.
    """
    def __init__(self, input_size: int, embed_dim: int):
        super().__init__()
        hidden = max(embed_dim, input_size // 4)
        # Cap hidden size to avoid huge layers for long features like raw/autocorr
        hidden = min(hidden, 512)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DynamicMultiBranchNet(nn.Module):
    """
    Dynamic multi-branch network.
    Branches are created at runtime based on input_sizes dict.
    Each branch compresses its feature to embed_dim.
    All embeddings are concatenated → classifier.
    """

    def __init__(self,
        input_sizes: dict[str, int],
        embed_dim: int,
        n_classes: int,
        p_drop: float
    ):
        super().__init__()
        self.feature_names = list(input_sizes.keys())

        # Create one branch per feature
        self.branches = nn.ModuleDict({
            name: _Branch(size, embed_dim)
            for name, size in input_sizes.items()
        })

        # Classifier takes concatenated embeddings from all branches
        total_embed = embed_dim * len(input_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(total_embed, total_embed // 2),
            nn.ReLU(),
            nn.Linear(total_embed // 2, n_classes),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = [
            self.branches[name](inputs[name])
            for name in self.feature_names
        ]
        combined = torch.cat(embeddings, dim=1)
        return self.classifier(combined)


class DynamicMultiBranchFFNNModel(TorchNNBase):
    """
    Dynamic multi-branch FFNN.

    Automatically creates one branch per feature in required_inputs.
    Each feature is processed at its natural size — no padding.
    Works with any combination of features from FEATURE_REGISTRY.

    Overrides _core_train/_core_infer/_core_avg_val_loss to pass
    features as separate tensors (dict) instead of one stacked tensor.

    Change required_inputs to use different features — branches
    are created automatically at the right input size.
    """

    # ── Change this to use different features ──────────────────
    required_inputs = ["raw", "pitchtrack", "autocorr", "autoencoder_latent"]
    # ───────────────────────────────────────────────────────────

    def __init__(self, training_options: dict):
        TorchNNBase.__init__(self, training_options)
        self.embed_dim = int(training_options.get("embed_dim", 64))
        self._net = None  # built lazily in _core_train

    def build(self):
        pass  # built lazily once we know feature sizes

    def _get_input_sizes(self, trials: list[EEGTrial]) -> dict[str, int]:
        """Determine input size for each feature from the first trial."""
        sizes = {}
        trial = trials[0]
        for name in self.required_inputs:
            if name == "raw":
                sizes[name] = len(np.array(trial.data).flatten())
            else:
                sizes[name] = len(np.array(trial.features[name]).flatten())
        return sizes

    def _build_net(self, trials: list[EEGTrial]):
        """Build the network once we know feature sizes."""
        input_sizes = self._get_input_sizes(trials)
        n_classes   = self.subject.num_categories
        p_drop      = float(self.training_options.get("p_drop", 0.1))

        print(f"DynamicMultiBranchFFNN | Building branches:")
        for name, size in input_sizes.items():
            print(f"  {name}: {size} → {self.embed_dim}")

        self._net = _DynamicMultiBranchNet(
            input_sizes=input_sizes,
            embed_dim=self.embed_dim,
            n_classes=n_classes,
            p_drop=p_drop,
        ).to(self.device)

    def _get_feature_tensors(self, trials: list[EEGTrial]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Extract each feature as a separate tensor at its natural size."""
        feats = {}
        for name in self.required_inputs:
            if name == "raw":
                arrays = [np.array(t.data, dtype=np.float32).flatten() for t in trials]
            else:
                arrays = [np.array(t.features[name], dtype=np.float32).flatten() for t in trials]
            feats[name] = torch.tensor(np.stack(arrays), dtype=torch.float32).to(self.device)

        labels = torch.tensor(
            [t.enumerated_label for t in trials], dtype=torch.long
        ).to(self.device)
        return feats, labels

    def _core_train(self, *,
        trials,
        validation_trials,
        num_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        min_delta,
        patience
    ):
        # Build network on first call
        self._build_net(trials)
        self.model = self._net  # set self.model so _store_best/_restore_best work

        optimizer = optim.AdamW(
            self._net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Handle validation split
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
                idx = indices[start:start + batch_size]
                batch_feats = {name: train_feats[name][idx] for name in self.required_inputs}
                batch_labels = train_labels[idx]

                optimizer.zero_grad(set_to_none=True)
                logits = self._net(batch_feats)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            if must_validate:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(val_feats)
                    val_loss = criterion(val_logits, val_labels).item()
                self._net.train()
                self._record_loss(val_loss, self._net)
                if not self._should_continue():
                    self._restore_best()
                    break

    def _core_infer(self, *,
        trials: list[EEGTrial],
        batch_size: int
    ) -> list[dict[int, float]]:

        feats, _ = self._get_feature_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(feats)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        return [
            {i: float(p) for i, p in enumerate(row)}
            for row in probs
        ]

    def _core_avg_val_loss(self, *,
        trials: list[EEGTrial],
        batch_size: int
    ) -> float:

        feats, labels = self._get_feature_tensors(trials)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(feats)
            loss   = nn.CrossEntropyLoss()(logits, labels).item()
        self._net.train()
        return loss