from .utils import TorchNNBase

import torch
import torch.nn as nn


class _MultiBranchCNN1D(nn.Module):
    """
    Two-branch CNN: one branch for raw waveform, one for pitch track.
    Each branch pools to a fixed-size embedding, then they're concatenated
    and passed to a classifier head.

    Expects inputs dict: {"raw": (B, T_raw), "pitchtrack": (B, T_pitch)}
    """

    def __init__(self, n_classes: int = 4, p_drop: float = 0.1):
        super().__init__()

        self.raw_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=251, padding=125, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(64, 128, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.pitch_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, padding=25, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # raw_branch outputs 64-dim, pitch_branch outputs 32-dim
        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(64 + 32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2, T) — raw is channel 0, pitch is channel 1
        # pitch is padded to raw length by extract_features
        x_raw   = x[:, 0, :]  # (B, T_raw)
        x_pitch = x[:, 1, :]  # (B, T_pitch) — padded but AdaptiveAvgPool handles it

        x_raw   = x_raw.unsqueeze(1)    # (B, 1, T_raw)
        x_pitch = x_pitch.unsqueeze(1)  # (B, 1, T_pitch)

        raw_feat   = self.raw_branch(x_raw)
        pitch_feat = self.pitch_branch(x_pitch)

        combined = torch.cat([raw_feat, pitch_feat], dim=1)
        return self.classifier(combined)


class MultiBranchCNNModel(TorchNNBase):
    required_inputs = ["raw", "pitchtrack"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", self.subject.num_categories if self.subject else 4))
        p_drop    = float(self.training_options.get("p_drop", 0.1))
        self.model = _MultiBranchCNN1D(n_classes=n_classes, p_drop=p_drop).to(self.device)
