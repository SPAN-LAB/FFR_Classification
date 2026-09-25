import torch
from torch import nn

from .utils import TorchNNBase


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden_size: int,
        frontend_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_size < 1 or frontend_channels < 4 or num_layers < 1:
            raise ValueError("Model dimensions and layer count must be positive")

        first_channels = max(frontend_channels // 2, 4)
        self.frontend = nn.Sequential(
            nn.Conv1d(
                1,
                first_channels,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.GroupNorm(1, first_channels),
            nn.GELU(),
            nn.Conv1d(
                first_channels,
                frontend_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(1, frontend_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
        )
        self.lstm = nn.LSTM(
            input_size=frontend_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.normalization = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self._initialize_forget_gates()

    def _initialize_forget_gates(self) -> None:
        for name, parameter in self.lstm.named_parameters():
            if "bias_ih" not in name:
                continue
            gate_size = parameter.shape[0] // 4
            with torch.no_grad():
                parameter[gate_size : 2 * gate_size].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(
                "Expected raw waveforms shaped [batch, time], "
                "[batch, 1, time], or [batch, time, 1]"
            )

        features = self.frontend(x).transpose(1, 2)
        sequence, _ = self.lstm(features)
        pooled = sequence.mean(dim=1)
        pooled = self.dropout(self.normalization(pooled))
        return self.classifier(pooled)


class RNN_model(TorchNNBase):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)

    def build(self):
        options = self.training_options or {}
        num_classes = int(
            options.get(
                "n_classes",
                self.subject.num_categories if self.subject else 4,
            )
        )
        self.model = CNN_LSTM(
            num_classes=num_classes,
            hidden_size=int(options.get("hidden_size", 128)),
            frontend_channels=int(options.get("frontend_channels", 32)),
            num_layers=int(options.get("num_layers", 1)),
            dropout=float(options.get("p_drop", options.get("dropout", 0.2))),
        ).to(self.device)
