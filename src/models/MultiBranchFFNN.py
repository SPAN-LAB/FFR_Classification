from math import floor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .utils.torchnn_base import TorchNNBase
from ..core import EEGTrial


class _Branch(nn.Module):
    def __init__(self, input_size: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _MultiBranchFFNN(nn.Module):
    """
    One branch per feature, all compressing to embed_dim.
    Branches are concatenated then passed to a classifier.
    """
    def __init__(self, input_sizes: dict[str, int], embed_dim: int, n_classes: int, p_drop: float):
        super().__init__()
        self.feature_names = list(input_sizes.keys())
        self.branches = nn.ModuleDict({
            name: _Branch(size, embed_dim)
            for name, size in input_sizes.items()
        })
        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(embed_dim * len(input_sizes), n_classes)
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = [self.branches[name](inputs[name]) for name in self.feature_names]
        return self.classifier(torch.cat(embeddings, dim=1))


class MultiBranchFFNNModel(TorchNNBase):
    """
    Multi-branch FFNN — each feature at its natural size, no padding.
    Overrides _core_train/_core_infer/_core_avg_val_loss to pass features
    as separate tensors instead of one stacked tensor.

    Usage:
        .extract_features(["pitchtrack", "autocorr"], concatenate=False)
        .evaluate_model("MultiBranchFFNN", training_options={...})
    """

    required_inputs = ["pitchtrack", "autocorr"]

    def __init__(self, training_options: dict):
        TorchNNBase.__init__(self, training_options)
        self.embed_dim = training_options.get("embed_dim", 64)

    def build(self):
        pass  # built lazily in _core_train once we know input sizes

    def _get_feature_tensors(self, trials: list[EEGTrial]) -> dict[str, torch.Tensor]:
        """Extract each feature as a separate tensor at its natural size."""
        result = {}
        for name in self.required_inputs:
            if name == "raw":
                arrays = [np.array(t.data, dtype=np.float32).flatten() for t in trials]
            else:
                arrays = [np.array(t.features[name], dtype=np.float32).flatten() for t in trials]
            result[name] = torch.tensor(np.stack(arrays), dtype=torch.float32).to(self.device)
        return result

    def _build_model(self, trials: list[EEGTrial]):
        """Build model once we know the actual input sizes."""
        tensors = self._get_feature_tensors(trials)
        input_sizes = {name: tensors[name].shape[1] for name in self.required_inputs}
        n_classes = self.subject.num_categories
        p_drop = float(self.training_options.get("p_drop", 0.1))
        self.model = _MultiBranchFFNN(
            input_sizes=input_sizes,
            embed_dim=self.embed_dim,
            n_classes=n_classes,
            p_drop=p_drop
        ).to(self.device)

    def _core_train(self, *, trials, validation_trials, num_epochs,
                batch_size, learning_rate, weight_decay, min_delta, patience):

        self._build_model(trials)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Handle validation split — same logic as TorchNNBase
        must_validate = validation_trials is not None and validation_trials != 0
        if must_validate and isinstance(validation_trials, float):
            from ..core.utils.sampling import sds2
            n_val = int(len(trials) * validation_trials)
            if n_val <= 0 or len(trials) - n_val <= 0:
                must_validate = False
            else:
                validation_trials = sds2(trials=trials, num_trials=n_val)
                for t in validation_trials:
                    trials.remove(t)

        def get_tensors(trial_list):
            feats = self._get_feature_tensors(trial_list)
            labels = torch.tensor(
                [t.enumerated_label for t in trial_list], dtype=torch.long
            ).to(self.device)
            return feats, labels

        train_feats, train_labels = get_tensors(trials)
        if must_validate:
            val_feats, val_labels = get_tensors(validation_trials)

        n = len(trials)
        self._reset_loss_trackers()
        self.model.train()

        for epoch in range(num_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                batch_feats = {name: train_feats[name][idx] for name in self.required_inputs}
                batch_labels = train_labels[idx]

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_feats)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            if must_validate:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_feats)
                    val_loss = criterion(val_logits, val_labels).item()
                self.model.train()
                self._record_loss(val_loss, self.model)
                if not self._should_continue():
                    self._restore_best()
                    break

    def _core_infer(self, *, trials, batch_size) -> list[dict[int, float]]:
        feats = self._get_feature_tensors(trials)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return [
            {i: float(p) for i, p in enumerate(row)}
            for row in probs
        ]

    def _core_avg_val_loss(self, *, trials, batch_size) -> float:
        feats = self._get_feature_tensors(trials)
        labels = torch.tensor(
            [t.enumerated_label for t in trials], dtype=torch.long
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(feats)
            loss = nn.CrossEntropyLoss()(logits, labels).item()
        self.model.train()
        return loss