import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

from .utils import TorchNNBase


class _MaxNormParametrize(nn.Module):
    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.renorm(p=2, dim=0, maxnorm=self.max_norm)


class _Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=1)
        register_parametrization(self, "weight", _MaxNormParametrize(max_norm))


class _EEGNetNet(nn.Module):
    """
    EEGNet v4 from Lawhern et al. (2018). Expects input [B, n_chans, T].

    Architecture:
        Temporal conv (learned filter bank)
        Depthwise spatial conv (per-filter spatial projection)
        BN + ELU + AvgPool + Dropout
        Depthwise-separable conv (depth + pointwise)
        BN + ELU + AvgPool + Dropout
        Conv classifier
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        depthwise_kernel_length: int = 16,
        pool1_kernel_size: int = 4,
        pool2_kernel_size: int = 8,
        conv_spatial_max_norm: float = 1.0,
        drop_prob: float = 0.25,
        batch_norm_momentum: float = 0.01,
        batch_norm_eps: float = 1e-3,
    ):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=batch_norm_momentum, eps=batch_norm_eps),
            _Conv2dWithConstraint(F1, F1 * D, (n_chans, 1), groups=F1, bias=False, max_norm=conv_spatial_max_norm),
            nn.BatchNorm2d(F1 * D, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d((1, pool1_kernel_size)),
            nn.Dropout(p=drop_prob),
            nn.Conv2d(F1 * D, F1 * D, (1, depthwise_kernel_length), groups=F1 * D, padding=(0, depthwise_kernel_length // 2), bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ELU(),
            nn.AvgPool2d((1, pool2_kernel_size)),
            nn.Dropout(p=drop_prob),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            out_shape = self.features(dummy).shape
            n_virtual_chans, n_out_time = out_shape[2], out_shape[3]

        self.classifier = nn.Conv2d(F2, n_outputs, (n_virtual_chans, n_out_time))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).unsqueeze(1)  # [B, T] → [B, 1, 1, T]
        elif x.ndim == 3:
            x = x.unsqueeze(1)               # [B, n_chans, T] → [B, 1, n_chans, T]
        x = self.features(x)
        x = self.classifier(x)
        return x.flatten(1)                  # [B, n_outputs, 1, 1] → [B, n_outputs]


class EEGNetModel(TorchNNBase):
    required_inputs = ["raw"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)

    def build(self) -> None:
        if self.subject is None:
            raise ValueError("EEGNet requires a subject before it can be built.")
        options = self.training_options or {}
        n_chans   = int(options.get("n_chans", 1))
        n_times   = int(options.get("n_times", self.subject.trial_size))
        n_classes = int(options.get("n_classes", self.subject.num_categories))
        drop_prob = float(options.get("drop_prob", 0.25))
        self.model = _EEGNetNet(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_classes,
            drop_prob=drop_prob,
        ).to(self.device)
