from .utils import TorchNNBase
from ..core.eeg_trial import EEGTrial

import torch
import torch.nn as nn


class _ConvTransformer1D(nn.Module):
    """
    Patch-Transformer encoder for 1D FFR signals (practical compute).

    Expects input [B, T].
    Converts the waveform into "patch tokens" using a strided Conv1d (patch_size),
    adds positional embeddings, prepends CLS, runs Transformer encoder, returns logits [B, n_classes].

    This keeps a transformer backbone but makes runtime/memory manageable because attention
    runs on ~T/patch_size tokens instead of T.
    """

    def __init__(
        self,
        n_classes: int = 4,
        d_model: int = 192,
        n_heads: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        max_len_reduced: int = 8192,   # still "max T" for compatibility
        patch_size: int = 8,           # key knob: 4/8/16
        p_drop: float = 0.1,
    ):
        super().__init__()

        self.max_len_reduced = max_len_reduced
        self.d_model = d_model
        self.patch_size = patch_size

        # Patch embedding: [B, T] -> [B, d_model, N] where N ~= ceil(T / patch_size)
        # This is the main compute saver.
        self.patch_embed = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Positional embeddings over patch tokens (not raw timesteps)
        # Allocate enough for the maximum possible number of patches.
        self.max_patches = (max_len_reduced + patch_size - 1) // patch_size
        self.pos_embedding = nn.Embedding(self.max_patches, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=p_drop,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        if x.ndim != 2:
            raise ValueError(f"Expected input of shape [B, T], got {x.shape}")

        B, T = x.shape
        if T > self.max_len_reduced:
            raise ValueError(
                f"Seq len {T} exceeds max_len_reduced={self.max_len_reduced}. "
                "Increase max_len_reduced."
            )

        # Patchify
        x = x.unsqueeze(1)            # [B, 1, T]
        x = self.patch_embed(x)       # [B, d_model, N]
        x = x.transpose(1, 2)         # [B, N, d_model]
        B, N, _ = x.shape

        if N > self.max_patches:
            raise ValueError(
                f"Num patches {N} exceeds max_patches={self.max_patches}. "
                "Increase max_len_reduced or patch_size."
            )

        # Positional embeddings
        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)  # [B, N]
        x = x + self.pos_embedding(pos_ids)  # [B, N, d_model]

        # CLS + encode
        cls = self.cls_token.expand(B, 1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)         # [B, N+1, d_model]

        x = self.encoder(x)        # [B, N+1, d_model]
        x = self.final_norm(x)

        cls_out = x[:, 0, :]       # [B, d_model]
        cls_out = self.dropout(cls_out)
        return self.fc(cls_out)    # [B, n_classes]


class ConvTransformerModel(TorchNNBase):
    """
    TorchNNBase wrapper so this drops into your pipeline like CNNModel.
    (Class name kept for compatibility; internals are a patch-based Transformer.)
    """

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", 4))
        p_drop = float(self.training_options.get("p_drop", 0.1))

        d_model = int(self.training_options.get("d_model", 192))
        n_heads = int(self.training_options.get("n_heads", 6))
        num_layers = int(self.training_options.get("num_layers", 4))
        dim_ff = int(self.training_options.get("dim_feedforward", 768))

        max_len_reduced = int(self.training_options.get("max_len_reduced", 8192))
        patch_size = int(self.training_options.get("patch_size", 8))

        self.model = _ConvTransformer1D(
            n_classes=n_classes,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            max_len_reduced=max_len_reduced,
            patch_size=patch_size,
            p_drop=p_drop,
        ).to(self.device)

