"""
SPAN Lab - FFR Classification

Filename: DynamicMultiInputTransformer.py
Author(s): Cj
Description: A dynamic multi-input patch transformer that automatically creates
             one patch embedder per feature based on required_inputs.
             Each feature gets its own PatchEmbedder + modality embedding.
             All tokens concatenated → shared Transformer → CLS → classifier.

             Works with any combination of features including autoencoder_latent.
             AE features (128) → 16 tokens with patch_size=8.
             Raw/autocorr (3277) → ~410 tokens with patch_size=8.
             Pitch (265) → ~34 tokens with patch_size=8.

To use different features — just change required_inputs:
    required_inputs = ["raw"]
    required_inputs = ["raw", "pitchtrack"]
    required_inputs = ["raw", "pitchtrack", "autocorr", "autoencoder_latent"]

Usage in demo.py:
    .extract_features(["pitchtrack", "autocorr", "autoencoder_latent"])
    .evaluate_model("DynamicMultiInputTransformer", training_options={
        "num_epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.00005,
        "weight_decay": 0.05,
        "patience": 30,
        "min_delta": 0.0005,
        "patch_size": 8,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
        "max_tokens": 2048,
    })
"""

import numpy as np
import torch
import torch.nn as nn

from .utils.torchnn_base import TorchNNBase
from ..core.eeg_trial import EEGTrial
from ..core.utils.sampling import sds2
from torch.utils.data import DataLoader, TensorDataset


class _PatchEmbedder(nn.Module):
    """
    Converts a 1D signal into patch tokens.
    Signal (B, T) → tokens (B, N, d_model) where N = T // patch_size
    """
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)      # (B, 1, T)
        x = self.embed(x)       # (B, d_model, N)
        x = x.transpose(1, 2)  # (B, N, d_model)
        return x


class _DynamicMultiInputTransformerNet(nn.Module):
    """
    Dynamic multi-input transformer.
    Creates one PatchEmbedder per feature automatically.
    All tokens from all features are concatenated → shared Transformer → CLS → classifier.
    """

    def __init__(self,
        feature_names: list[str],
        feature_sizes: dict[str, int],
        n_classes: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        patch_size: int,
        max_tokens: int,
        p_drop: float,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.patch_size    = patch_size
        self.d_model       = d_model

        # One patch embedder per feature
        self.patch_embedders = nn.ModuleDict({
            name: _PatchEmbedder(patch_size, d_model)
            for name in feature_names
        })

        # Modality embedding — distinguishes which feature each token came from
        self.modality_embeddings = nn.Embedding(len(feature_names), d_model)

        # Positional embedding over all tokens
        self.pos_embedding = nn.Embedding(max_tokens + 1, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=p_drop,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p_drop)
        self.fc         = nn.Linear(d_model, n_classes)

        # Log token counts per feature
        print("DynamicMultiInputTransformer | Token counts per feature:")
        total = 0
        for name, size in feature_sizes.items():
            n_tokens = size // patch_size
            total += n_tokens
            print(f"  {name}: {size} pts → {n_tokens} tokens (patch_size={patch_size})")
        print(f"  Total tokens: {total} + 1 CLS = {total+1}")

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        B = next(iter(inputs.values())).shape[0]
        all_tokens = []

        for i, name in enumerate(self.feature_names):
            feat = inputs[name]  # (B, T)

            # Embed into patches
            tokens = self.patch_embedders[name](feat)  # (B, N_i, d_model)
            N_i = tokens.shape[1]

            # Add modality embedding
            mod_embed = self.modality_embeddings(
                torch.full((B, N_i), i, dtype=torch.long, device=feat.device)
            )
            tokens = tokens + mod_embed
            all_tokens.append(tokens)

        # Concatenate all feature tokens
        tokens  = torch.cat(all_tokens, dim=1)   # (B, N_total, d_model)
        N_total = tokens.shape[1]

        # Add positional embeddings
        pos_ids = torch.arange(N_total, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens  = tokens + self.pos_embedding(pos_ids)

        # Prepend CLS token
        cls    = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N_total+1, d_model)

        # Transformer
        tokens = self.encoder(tokens)
        tokens = self.final_norm(tokens)

        # CLS → classifier
        cls_out = tokens[:, 0, :]
        cls_out = self.dropout(cls_out)
        return self.fc(cls_out)


class DynamicMultiInputTransformerModel(TorchNNBase):
    """
    Dynamic multi-input patch transformer.
    Change required_inputs to use different features — patch embedders
    are created automatically at the right size.
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
                arr = np.array(trial.data, dtype=np.float32).flatten()
            else:
                arr = np.array(trial.features[name], dtype=np.float32).flatten()
            patch_size = int(self.training_options.get("patch_size", 8))
            # Trim to nearest multiple of patch_size
            trimmed_len = (len(arr) // patch_size) * patch_size
            sizes[name] = trimmed_len
        return sizes

    def _build_net(self, trials: list[EEGTrial]):
        feature_sizes  = self._get_feature_sizes(trials)
        n_classes      = self.subject.num_categories
        d_model        = int(self.training_options.get("d_model", 128))
        n_heads        = int(self.training_options.get("n_heads", 4))
        num_layers     = int(self.training_options.get("num_layers", 3))
        dim_feedforward= int(self.training_options.get("dim_feedforward", 512))
        patch_size     = int(self.training_options.get("patch_size", 8))
        max_tokens     = int(self.training_options.get("max_tokens", 2048))
        p_drop         = float(self.training_options.get("p_drop", 0.1))

        self._net = _DynamicMultiInputTransformerNet(
            feature_names=self.required_inputs,
            feature_sizes=feature_sizes,
            n_classes=n_classes,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            patch_size=patch_size,
            max_tokens=max_tokens,
            p_drop=p_drop,
        ).to(self.device)
        self.model = self._net

    def _get_feature_tensors(self, trials: list[EEGTrial]) -> tuple[dict, torch.Tensor]:
        patch_size = int(self.training_options.get("patch_size", 8))
        feats  = {}
        for name in self.required_inputs:
            if name == "raw":
                arrays = [np.array(t.data, dtype=np.float32).flatten() for t in trials]
            else:
                arrays = [np.array(t.features[name], dtype=np.float32).flatten() for t in trials]
            # Trim to nearest multiple of patch_size
            trimmed = [(a[: (len(a) // patch_size) * patch_size]) for a in arrays]
            feats[name] = torch.tensor(
                np.stack(trimmed), dtype=torch.float32
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

        import torch.optim as optim
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
                idx = indices[start:start + batch_size]
                batch_feats = {name: train_feats[name][idx] for name in self.required_inputs}
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