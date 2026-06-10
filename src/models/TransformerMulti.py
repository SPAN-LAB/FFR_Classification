"""
Description: Multi-input Patch Transformer for FFR classification.
             Each feature gets its own patch embedder and modality embedding.
             All tokens are concatenated and processed by a shared Transformer.
             Cross-feature attention happens naturally — no padding needed.

Architecture:
    Raw (3277)     → PatchEmbed_raw   → ~410 tokens + modality_raw
    Pitch (265)    → PatchEmbed_pitch → ~34 tokens  + modality_pitch
    Autocorr (3277)→ PatchEmbed_ac    → ~410 tokens + modality_ac
                   → concat → [CLS] + all tokens → Transformer → CLS → 4 classes

Usage in demo.py:
    .extract_features(["pitchtrack", "autocorr"], concatenate=False)
    .evaluate_model("MultiInputTransformer", training_options={
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "weight_decay": 0.1,
        "patience": 20,
        "min_delta": 0.001,
        "patch_size": 8,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
    })
"""

import torch
import torch.nn as nn
import numpy as np

from .utils import TorchNNBase
from .utils.index_tracked_dataset import IndexTrackedDataset
from torch.utils.data import DataLoader


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
        x = x.unsqueeze(1)          # (B, 1, T)
        x = self.embed(x)           # (B, d_model, N)
        x = x.transpose(1, 2)       # (B, N, d_model)
        return x


class _MultiInputTransformer1D(nn.Module):
    """
    Multi-input Patch Transformer.
    Each feature has its own patch embedder and modality embedding.
    All tokens are concatenated then processed by a shared Transformer encoder.
    """

    def __init__(
        self,
        feature_names: list[str],
        n_classes: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        patch_size: int = 8,
        max_tokens: int = 2048,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.d_model = d_model
        self.patch_size = patch_size

        # Separate patch embedder per feature
        self.patch_embedders = nn.ModuleDict({
            name: _PatchEmbedder(patch_size, d_model)
            for name in feature_names
        })

        # Learned modality embedding per feature — tells transformer which feature each token came from
        self.modality_embeddings = nn.Embedding(len(feature_names), d_model)

        # Positional embedding over all tokens
        self.pos_embedding = nn.Embedding(max_tokens + 1, d_model)  # +1 for CLS

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features, T) — stacked by IndexTrackedDataset
        Each channel is one feature, all padded to same length T.
        We split by channel, embed each separately, then concatenate tokens.
        """
        B = x.shape[0]
        all_tokens = []

        for i, name in enumerate(self.feature_names):
            feat = x[:, i, :]  # (B, T) — may include padding zeros for pitch

            # Embed into patches
            tokens = self.patch_embedders[name](feat)  # (B, N_i, d_model)
            N_i = tokens.shape[1]

            # Add modality embedding — tells transformer which feature these tokens are from
            mod_embed = self.modality_embeddings(
                torch.full((B, N_i), i, dtype=torch.long, device=x.device)
            )  # (B, N_i, d_model)
            tokens = tokens + mod_embed

            all_tokens.append(tokens)

        # Concatenate all feature tokens: (B, N_total, d_model)
        tokens = torch.cat(all_tokens, dim=1)
        N_total = tokens.shape[1]

        # Add positional embedding over all tokens
        pos_ids = torch.arange(N_total, device=x.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embedding(pos_ids)

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N_total+1, d_model)

        # Transformer encoder
        tokens = self.encoder(tokens)       # (B, N_total+1, d_model)
        tokens = self.final_norm(tokens)

        # CLS output → classifier
        cls_out = tokens[:, 0, :]           # (B, d_model)
        cls_out = self.dropout(cls_out)
        return self.fc(cls_out)             # (B, n_classes)


class MultiInputTransformerModel(TorchNNBase):
    """
    Multi-input Patch Transformer — drops into the pipeline like any other model.
    Each feature in required_inputs gets its own patch embedder and modality embedding.
    Cross-feature attention happens naturally inside the shared Transformer.

    required_inputs controls which features are used:
        ["raw"]                          → single input (same as ConvTransformer)
        ["raw", "pitchtrack"]            → raw + pitch
        ["raw", "autocorr", "pitchtrack"]→ all three features
    """

    required_inputs = ["raw", "autocorr", "pitchtrack"]

    def __init__(self, training_options: dict):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_classes      = int(self.training_options.get("n_classes", 4))
        p_drop         = float(self.training_options.get("p_drop", 0.1))
        d_model        = int(self.training_options.get("d_model", 128))
        n_heads        = int(self.training_options.get("n_heads", 4))
        num_layers     = int(self.training_options.get("num_layers", 3))
        dim_feedforward= int(self.training_options.get("dim_feedforward", 512))
        patch_size     = int(self.training_options.get("patch_size", 8))
        max_tokens     = int(self.training_options.get("max_tokens", 2048))

        self.model = _MultiInputTransformer1D(
            feature_names=self.required_inputs,
            n_classes=n_classes,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            patch_size=patch_size,
            max_tokens=max_tokens,
            p_drop=p_drop,
        ).to(self.device)