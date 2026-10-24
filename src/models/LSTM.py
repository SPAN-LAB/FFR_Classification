"""LSTM: the frozen EEGNeXFFR CNN-LSTM baseline, previously named RNN_V1.

The selected default uses a small LSTM and retains every frontend time step.
The earlier vanilla CNN-RNN remains available as architecture='conv_rnn'.
"""

import torch
from torch import nn

from .utils import TorchNNBase
from .EEGNeXFFR import _EEGNeXFFRNet, _LinearWithConstraint


class EEGNeXRecurrent(nn.Module):
    """EEGNeXFFR features plus a gated recurrent residual at every time step."""

    def __init__(
        self,
        *,
        n_times: int,
        num_classes: int,
        hidden_size: int = 16,
        pool_size: int = 4,
        dropout: float = 0.5,
        recurrent_type: str = "lstm",
        normalize_input: bool = False,
        kernel_size: int = 64,
        sinc_frontend: bool = False,
    ):
        super().__init__()
        if min(n_times, num_classes, hidden_size, pool_size, kernel_size) < 1:
            raise ValueError("Model dimensions must be positive")
        if recurrent_type not in {"rnn", "gru", "lstm"}:
            raise ValueError("recurrent_type must be rnn, gru, or lstm")
        self.input_normalization = (
            nn.GroupNorm(1, 1, affine=False) if normalize_input else nn.Identity()
        )
        cnn = _EEGNeXFFRNet(
            n_times=n_times,
            n_outputs=num_classes,
            avg_pool_block5=pool_size,
            drop_prob=dropout,
            kernel_block_1_2=kernel_size,
            sinc_frontend=sinc_frontend,
        )
        # Reuse the actual EEGNeXFFR blocks, removing only the final flatten.
        self.frontend = nn.Sequential(
            cnn.block_1,
            cnn.block_2,
            cnn.block_3,
            cnn.block_4,
            *list(cnn.block_5.children())[:-1],
        )
        self.num_steps = cnn.final_layer.in_features // 8
        recurrent_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[
            recurrent_type
        ]
        self.recurrent = recurrent_class(
            8, hidden_size, batch_first=True, bidirectional=False
        )
        self.projection = nn.Linear(hidden_size, 8, bias=False)
        # Begin with the direct CNN features; learn a recurrent correction.
        nn.init.zeros_(self.projection.weight)
        self.dropout = nn.Dropout(dropout)
        self.classifier = _LinearWithConstraint(
            8 * self.num_steps, num_classes, max_norm=0.25
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected single-channel waveforms")
        features = self.frontend(self.input_normalization(x).unsqueeze(2))
        features = features.squeeze(2).transpose(1, 2)  # [batch, time, 8]
        sequence, _ = self.recurrent(features)
        combined = features + self.dropout(self.projection(sequence))
        return self.classifier(combined.transpose(1, 2).flatten(1))


class ConvRNNBaseline(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden_size: int = 64,
        frontend_channels: int = 16,
        n_times: int = 4915,
        pool_size: int = 8,
        recurrent_type: str = "rnn",
        dropout: float = 0.2,
        channel_dropout: float = 0.1,
    ):
        super().__init__()
        if min(num_classes, hidden_size, frontend_channels, n_times, pool_size) < 1:
            raise ValueError("Model dimensions must be positive")
        if recurrent_type not in {"rnn", "gru", "lstm"}:
            raise ValueError("recurrent_type must be rnn, gru, or lstm")
        self.num_steps = n_times // (4 * pool_size)
        if self.num_steps < 1:
            raise ValueError("Trial is shorter than the frontend pooling window")

        # Normalize each waveform independently without changing the source data.
        self.input_normalization = nn.GroupNorm(1, 1, affine=False)

        # EEGNeXFFR-inspired temporal filterbank, depthwise expansion and dilated
        # convolutions. Only local pooling is used: the time axis survives.
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=65, padding=32, bias=False),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 32, kernel_size=65, padding=32, bias=False),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=17, padding=8, groups=32, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout1d(channel_dropout),
            nn.Conv1d(64, 32, kernel_size=17, dilation=2, padding=16, bias=False),
            nn.BatchNorm1d(32),
            nn.Conv1d(
                32,
                frontend_channels,
                kernel_size=17,
                dilation=4,
                padding=32,
                bias=False,
            ),
            nn.BatchNorm1d(frontend_channels),
            nn.ELU(),
            nn.AvgPool1d(pool_size),
            nn.Dropout1d(channel_dropout),
        )
        recurrent_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[
            recurrent_type
        ]
        self.recurrent = recurrent_class(
            input_size=frontend_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        # Keep every frontend time step. The head is sized from the trial length
        # and pooling factors, so increasing resolution cannot silently pool it away.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_steps, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(
                "Expected waveforms shaped [batch, time] or [batch, 1, time]"
            )

        features = self.frontend(self.input_normalization(x)).transpose(1, 2)
        sequence, _ = self.recurrent(features)
        return self.classifier(sequence.transpose(1, 2))


class RNN_model(TorchNNBase):
    def reset_seed(self):
        seed = int((self.training_options or {}).get("seed", 0))
        torch.manual_seed(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def get_validation_ratio(self) -> float:
        """Train on every training-fold trial for all configured epochs."""
        return 0.0

    def build(self):
        options = self.training_options or {}
        num_classes = int(
            options.get(
                "n_classes",
                self.subject.num_categories if self.subject else 4,
            )
        )
        architecture = options.get("architecture", "eegnex_residual")
        if architecture == "eegnex_residual":
            self.model = EEGNeXRecurrent(
                n_times=int(
                    options.get(
                        "n_times", self.subject.trial_size if self.subject else 4915
                    )
                ),
                num_classes=num_classes,
                hidden_size=int(options.get("hidden_size", 16)),
                pool_size=int(options.get("pool_size", 4)),
                dropout=float(options.get("p_drop", 0.5)),
                recurrent_type=options.get("recurrent_type", "lstm"),
                normalize_input=bool(options.get("normalize_input", False)),
                kernel_size=int(options.get("kernel_size", 64)),
                sinc_frontend=bool(options.get("sinc_frontend", False)),
            ).to(self.device)
            return
        if architecture != "conv_rnn":
            raise ValueError("architecture must be eegnex_residual or conv_rnn")
        self.model = ConvRNNBaseline(
            num_classes=num_classes,
            hidden_size=int(options.get("hidden_size", 64)),
            frontend_channels=int(options.get("frontend_channels", 16)),
            n_times=int(
                options.get(
                    "n_times", self.subject.trial_size if self.subject else 4915
                )
            ),
            pool_size=int(options.get("pool_size", 8)),
            recurrent_type=options.get("recurrent_type", "rnn"),
            dropout=float(options.get("p_drop", 0.2)),
            channel_dropout=float(options.get("channel_dropout", 0.1)),
        ).to(self.device)
