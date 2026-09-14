"""
SPAN Lab - FFR Classification

Filename: EEGNeXFFR.py
Description: EEGNeX, modified for SINGLE-CHANNEL FFR. Discoverable as "EEGNeXFFR".

    Motivation
    ----------
    EEGNeX is the strongest model on this data (52.8% single-subject, ~84/66/70%
    subaverage=5 on 4T1002/4T1004/4T1005). Its power comes from: two FULL temporal
    filterbank convs, a depth-multiplier expansion, dilated temporal convs, and a head
    that FLATTENS the whole time axis into a max-norm linear classifier. (My earlier v2
    failed because it replaced the full convs with separable ones and replaced the
    flatten with aggressive pooling — both threw away EEGNeX's capacity. This model does
    NOT repeat that; the body is EEGNeX verbatim.)

    The one single-channel change
    -----------------------------
    EEGNeX block 3 is a depthwise *spatial* conv with kernel (n_chans, 1). On a single
    channel that kernel is (1, 1) — it performs NO spatial integration; it is just a
    1x1 channel expansion. On single-channel FFR the spatial axis is dead weight, so we
    redirect that block into a depthwise *temporal* conv with kernel (1, k_block_3): it
    still expands channels (depth multiplier) AND now does real temporal work, which is
    where all the single-channel signal lives. Everything else is unchanged.

    Optional hypothesis knob
    ------------------------
    sinc_frontend (default False) replaces block 1's learned temporal conv with an
    F0/harmonic-biased learnable sinc filterbank (NaN-safe: positive-lag build, analytic
    centre tap). With it OFF this model reproduces EEGNeX (+ the temporal-block-3 tweak)
    and should land near EEGNeX's numbers; with it ON you get a clean test of whether an
    F0-biased front-end helps the proven EEGNeX body — the experiment for the paper.

    Pipeline contract: required_inputs = ["raw"]; trial.data is 1-D (T,) -> batched
    (B, T); forward expands to [B, 1, n_chans=1, T] like EEGNeX.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from .utils import TorchNNBase


# ---------------------------------------------------------------------------
# Max-norm helpers (verbatim from EEGNeX.py).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Optional NaN-safe F0/harmonic-biased sinc front-end, shaped as a Conv2d so it is a
# drop-in for EEGNeX block 1. Produces out_channels temporal filters with kernel (1, K).
# ---------------------------------------------------------------------------
class _SincConv2d(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16384,
                 min_low_hz: float = 50.0, min_band_hz: float = 20.0,
                 init_low_hz: float = 70.0, init_high_hz: float = 1500.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = torch.logspace(math.log10(init_low_hz), math.log10(init_high_hz), out_channels)
        self.low_hz_ = nn.Parameter(low_hz.view(-1, 1))
        self.band_hz_ = nn.Parameter((0.25 * low_hz).view(-1, 1))

        n = (kernel_size - 1) // 2
        self.register_buffer("t_right", (torch.arange(1, n + 1).float() / sample_rate).view(1, -1))
        self.register_buffer(
            "window", 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size) / kernel_size)
        )

    def _build_filters(self) -> torch.Tensor:
        nyq = self.sample_rate / 2.0
        low = torch.clamp(self.min_low_hz + torch.abs(self.low_hz_), max=nyq - self.min_band_hz - 1.0)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), max=nyq)
        band = high - low
        t = self.t_right
        left = (torch.sin(2 * math.pi * high * t) - torch.sin(2 * math.pi * low * t)) / (math.pi * t)
        kernel = torch.cat([left.flip(dims=[1]), 2 * band, left], dim=1) / (2 * band)
        kernel = kernel * self.window.view(1, -1)
        return kernel.view(self.out_channels, 1, 1, self.kernel_size)  # (out, 1, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._build_filters().to(x.device), padding=(0, self.kernel_size // 2))


# ---------------------------------------------------------------------------
# Network. Body is EEGNeX; block 3 is temporal (single-channel change); block 1 optional sinc.
# ---------------------------------------------------------------------------
class _EEGNeXFFRNet(nn.Module):
    def __init__(
        self,
        n_chans: int = 1,
        n_times: int = 4915,
        n_outputs: int = 4,
        filter_1: int = 8,
        filter_2: int = 32,
        depth_multiplier: int = 2,
        kernel_block_1_2: int = 64,
        kernel_block_3: int = 16,        # NEW: temporal kernel for the (formerly spatial) block 3
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        drop_prob: float = 0.5,
        sinc_frontend: bool = False,
        init_low_hz: float = 70.0,
        init_high_hz: float = 1500.0,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
    ):
        super().__init__()
        filter_3 = filter_2 * depth_multiplier

        # Block 1: temporal filterbank — learned conv (EEGNeX) or F0-biased sinc.
        if sinc_frontend:
            self.block_1 = nn.Sequential(
                _SincConv2d(filter_1, kernel_block_1_2, init_low_hz=init_low_hz, init_high_hz=init_high_hz),
                nn.BatchNorm2d(filter_1),
            )
        else:
            self.block_1 = nn.Sequential(
                nn.Conv2d(1, filter_1, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
                nn.BatchNorm2d(filter_1),
            )

        # Block 2: second temporal filterbank (EEGNeX, unchanged).
        self.block_2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(filter_2),
        )

        # Block 3: SINGLE-CHANNEL CHANGE. EEGNeX used a depthwise SPATIAL conv (n_chans,1)
        # which is a no-op (1,1) on one channel. Here it is a depthwise TEMPORAL conv
        # (1, kernel_block_3) that still expands filter_2 -> filter_3 but now does real
        # temporal work. Kept depthwise + max-norm like the original.
        self.block_3 = nn.Sequential(
            _Conv2dWithConstraint(
                filter_2, filter_3,
                kernel_size=(1, kernel_block_3),
                padding="same",
                groups=filter_2,
                bias=False,
                max_norm=max_norm_conv,
            ),
            nn.BatchNorm2d(filter_3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4), padding=(0, 1)),
            nn.Dropout(p=drop_prob),
        )

        # Blocks 4-5: dilated temporal convs (EEGNeX, unchanged — full convs).
        self.block_4 = nn.Sequential(
            nn.Conv2d(filter_3, filter_2, kernel_size=(1, kernel_block_4),
                      dilation=(1, dilation_block_4), padding="same", bias=False),
            nn.BatchNorm2d(filter_2),
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(filter_2, filter_1, kernel_size=(1, kernel_block_5),
                      dilation=(1, dilation_block_5), padding="same", bias=False),
            nn.BatchNorm2d(filter_1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block5), padding=(0, 1)),
            nn.Dropout(p=drop_prob),
            nn.Flatten(),
        )

        # Head: flatten the full time axis into a max-norm linear (EEGNeX — the part that
        # matters). in_features computed by a dummy forward so it adapts to n_times exactly.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            feat = self.block_5(self.block_4(self.block_3(self.block_2(self.block_1(dummy)))))
            in_features = feat.shape[1]
        self.final_layer = _LinearWithConstraint(in_features, n_outputs, max_norm=max_norm_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).unsqueeze(1)   # [B, T] -> [B, 1, 1, T]
        elif x.ndim == 3:
            x = x.unsqueeze(1)                # [B, n_chans, T] -> [B, 1, n_chans, T]
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.final_layer(x)


class EEGNeXFFRModel(TorchNNBase):
    required_inputs = ["raw"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        opt = self.training_options if isinstance(self.training_options, dict) else {}
        self.model = _EEGNeXFFRNet(
            n_chans=int(opt.get("n_chans", 1)),
            n_times=int(opt.get("n_times", 4915)),
            n_outputs=int(opt.get("n_classes", 4)),
            kernel_block_3=int(opt.get("kernel_block_3", 16)),
            drop_prob=float(opt.get("drop_prob", 0.5)),
            sinc_frontend=bool(opt.get("sinc_frontend", False)),
            init_low_hz=float(opt.get("init_low_hz", 70.0)),
            init_high_hz=float(opt.get("init_high_hz", 1500.0)),
        ).to(self.device)
