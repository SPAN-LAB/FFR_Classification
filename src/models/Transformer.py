from .utils import TorchNNBase
from ..core.eeg_trial import EEGTrial

import torch
import torch.nn as nn


class _ConvTransformer1D(nn.Module):
    """
    CNN + Transformer encoder for 1D FFR signals.

    Expects input of shape [B, T].
    Uses a small Conv1d front-end to reduce T, then a Transformer encoder.
    Returns logits of shape [B, n_classes].
    """

    def __init__(
        self,
        n_classes: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        max_len_reduced: int = 1024,
        p_drop: float = 0.1,
    ):
        super().__init__()

        self.max_len_reduced = max_len_reduced
        self.d_model = d_model

        # --- CNN front-end: [B, T] -> [B, d_model, T_reduced] ---
        # For T=4000, stride 2 + pool 2 ≈ T/4 = 1000.
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=9,
                padding=4,
                stride=2,  # T -> T/2
                bias=False,
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # T/2 -> T/4
        )

        # --- Positional embeddings on reduced time axis ---
        self.pos_embedding = nn.Embedding(max_len_reduced, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=p_drop,
            batch_first=True,  # [B, T_red, d_model]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for sequence-level classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        if x.ndim != 2:
            raise ValueError(f"Expected input of shape [B, T], got {x.shape}")

        B, T = x.shape

        # --- CNN front-end ---
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv(x)  # [B, d_model, T_reduced]
        x = x.transpose(1, 2)  # [B, T_reduced, d_model]
        B, T_red, _ = x.shape

        if T_red > self.max_len_reduced:
            raise ValueError(
                f"Reduced seq len {T_red} exceeds max_len_reduced={self.max_len_reduced}. "
                "Increase max_len_reduced or use more pooling."
            )

        # --- Positional embeddings ---
        pos_ids = torch.arange(T_red, device=x.device).unsqueeze(0).expand(B, T_red)
        pos_emb = self.pos_embedding(pos_ids)  # [B, T_red, d_model]
        x = x + pos_emb  # [B, T_red, d_model]

        # --- Prepend CLS token ---
        cls = self.cls_token.expand(B, 1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, T_red+1, d_model]

        # --- Transformer encoder ---
        x = self.encoder(x)  # [B, T_red+1, d_model]

        # --- Classification head (use CLS) ---
        cls_out = x[:, 0, :]  # [B, d_model]
        cls_out = self.dropout(cls_out)
        logits = self.fc(cls_out)  # [B, n_classes]

        return logits


class ConvTransformerModel(TorchNNBase):
    """
    TorchNNBase wrapper so this drops into your pipeline like CNNModel.
    """

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", 4))
        p_drop = float(self.training_options.get("p_drop", 0.1))

        d_model = int(self.training_options.get("d_model", 64))
        n_heads = int(self.training_options.get("n_heads", 4))
        num_layers = int(self.training_options.get("num_layers", 3))
        dim_ff = int(self.training_options.get("dim_feedforward", 256))
        max_len_reduced = int(self.training_options.get("max_len_reduced", 1024))

        self.model = _ConvTransformer1D(
            n_classes=n_classes,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            max_len_reduced=max_len_reduced,
            p_drop=p_drop,
        ).to(self.device)
