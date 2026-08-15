import math

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


class _LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        register_parametrization(self, "weight", _MaxNormParametrize(max_norm))


class _EEGNeXNet(nn.Module):
    """
    EEGNeX from Chen et al. (2024). Expects input [B, n_chans, T].

    Architecture:
        Block 1-2: temporal convolutions (learned FIR-like filter bank)
        Block 3:   depthwise spatial conv + AvgPool + Dropout
        Block 4-5: dilated temporal convolutions + AvgPool + Dropout
        Classifier: max-norm linear
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int = 4,
        filter_1: int = 8,
        filter_2: int = 32,
        depth_multiplier: int = 2,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        drop_prob: float = 0.5,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
    ):
        super().__init__()

        filter_3 = filter_2 * depth_multiplier

        self.block_1 = nn.Sequential(
            nn.Conv2d(1, filter_1, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(filter_1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(filter_2),
        )

        self.block_3 = nn.Sequential(
            _Conv2dWithConstraint(
                filter_2, filter_3,
                kernel_size=(n_chans, 1),
                groups=filter_2,
                bias=False,
                max_norm=max_norm_conv,
            ),
            nn.BatchNorm2d(filter_3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4), padding=(0, 1)),
            nn.Dropout(p=drop_prob),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                filter_3, filter_2,
                kernel_size=(1, kernel_block_4),
                dilation=(1, dilation_block_4),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filter_2),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                filter_2, filter_1,
                kernel_size=(1, kernel_block_5),
                dilation=(1, dilation_block_5),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filter_1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block5), padding=(0, 1)),
            nn.Dropout(p=drop_prob),
            nn.Flatten(),
        )

        t3 = math.floor((n_times + 2 - avg_pool_block4) / avg_pool_block4) + 1
        t5 = math.floor((t3 + 2 - avg_pool_block5) / avg_pool_block5) + 1
        in_features = filter_1 * t5

        self.final_layer = _LinearWithConstraint(in_features, n_outputs, max_norm=max_norm_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).unsqueeze(1)  # [B, T] → [B, 1, 1, T]
        elif x.ndim == 3:
            x = x.unsqueeze(1)               # [B, n_chans, T] → [B, 1, n_chans, T]
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.final_layer(x)


class EEGNeXModel(TorchNNBase):
    required_inputs = ["raw"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)

    def build(self) -> None:
        if self.subject is None:
            raise ValueError("EEGNeX requires a subject before it can be built.")
        options = self.training_options or {}
        n_chans   = int(options.get("n_chans", 1))
        n_times   = int(options.get("n_times", self.subject.trial_size))
        n_classes = int(options.get("n_classes", self.subject.num_categories))
        drop_prob = float(options.get("drop_prob", 0.5))
        self.model = _EEGNeXNet(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_classes,
            drop_prob=drop_prob,
        ).to(self.device)
