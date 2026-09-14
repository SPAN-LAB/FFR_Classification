import torch
import torch.nn as nn

from .utils import TorchNNBase


class _ResLayer(nn.Module):
    """Dilated Conv1d + BN + GELU + residual skip."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        batch_norm: bool,
        drop_prob: float,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding, bias=not batch_norm)
        self.bn   = nn.BatchNorm1d(out_ch) if batch_norm else nn.Identity()
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x)))) + self.skip(x)


class _GLULayer(nn.Module):
    """Dilated Conv1d with GLU activation (no BN, no residual).

    The main conv outputs 2*out_ch; one half gates the other via sigmoid.
    An optional context conv (small kernel) is added to the pre-gate output
    to let the gate attend to local temporal context.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        glu_context: int,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, 2 * out_ch, kernel_size, dilation=dilation, padding=padding)
        if glu_context > 0:
            ctx_pad = (glu_context - 1) // 2
            self.ctx_conv = nn.Conv1d(in_ch, 2 * out_ch, glu_context, padding=ctx_pad)
        else:
            self.ctx_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.ctx_conv is not None:
            h = h + self.ctx_conv(x)
        a, b = h.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class _ConvSequence(nn.Module):
    """Stack of residual dilated conv layers with periodic GLU layers."""

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        kernel_size: int,
        growth: float,
        dilation_growth: int,
        dilation_period: int,
        drop_prob: float,
        batch_norm: bool,
        glu: int,
        glu_context: int,
    ):
        super().__init__()
        layers = []
        in_ch = hidden_dim
        for i in range(depth):
            out_ch = int(hidden_dim * (growth ** i))
            dil    = dilation_growth ** (i % dilation_period)
            is_glu = (glu > 0) and ((i + 1) % glu == 0)
            if is_glu:
                layers.append(_GLULayer(in_ch, out_ch, kernel_size, dil, glu_context))
            else:
                layers.append(_ResLayer(in_ch, out_ch, kernel_size, dil, batch_norm, drop_prob))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.out_ch = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _BrainModuleNet(nn.Module):
    """
    BrainModule / SimpleConv from Défossez et al. (2023). Expects input [B, n_chans, T].

    Architecture:
        Input projection  1x1 Conv → hidden_dim
        ConvSequence      depth dilated residual blocks (every glu-th replaced by GLU block)
        Head              1x1 Conv → GELU → 1x1 Conv → temporal mean → logits
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int = 4,
        hidden_dim: int = 64,
        depth: int = 6,
        kernel_size: int = 3,
        growth: float = 1.0,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        conv_drop_prob: float = 0.0,
        dropout_input: float = 0.0,
        batch_norm: bool = True,
        glu: int = 2,
        glu_context: int = 1,
    ):
        super().__init__()
        self.input_drop = nn.Dropout(dropout_input) if dropout_input > 0 else nn.Identity()
        self.input_proj = nn.Conv1d(n_chans, hidden_dim, 1, bias=False)
        self.encoder    = _ConvSequence(
            hidden_dim, depth, kernel_size, growth,
            dilation_growth, dilation_period,
            conv_drop_prob, batch_norm, glu, glu_context,
        )
        enc_ch = self.encoder.out_ch
        self.head = nn.Sequential(
            nn.Conv1d(enc_ch, enc_ch, 1),
            nn.GELU(),
            nn.Conv1d(enc_ch, n_outputs, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)          # [B, T] → [B, 1, T]
        x = self.input_drop(x)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.head(x)
        return x.mean(dim=-1)           # temporal mean → [B, n_outputs]


class BrainModuleModel(TorchNNBase):
    required_inputs = ["raw"]

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.build()

    def build(self) -> None:
        n_chans    = int(self.training_options.get("n_chans",    1))
        n_classes  = int(self.training_options.get("n_classes",  4))
        hidden_dim = int(self.training_options.get("hidden_dim", 64))
        depth      = int(self.training_options.get("depth",      6))
        drop_prob  = float(self.training_options.get("drop_prob", 0.0))
        self.model = _BrainModuleNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            hidden_dim=hidden_dim,
            depth=depth,
            conv_drop_prob=drop_prob,
        ).to(self.device)
