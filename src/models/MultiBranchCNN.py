from .utils import TorchNNBase

import torch
import torch.nn as nn


_DEFAULT_INPUTS = ["raw", "pitchtrack"]
_FEATURE_INPUTS = {"pitchtrack", "autocorr", "zerocrossing"}


def _parse_feature_inputs(value) -> list[str]:
    if value is None or value == "":
        return list(_DEFAULT_INPUTS)
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",")]
    inputs = [name for name in value if name]
    if len(inputs) < 2:
        raise ValueError("MultiBranchCNN needs at least two inputs, e.g. raw,pitchtrack.")
    invalid = [name for name in inputs if name != "raw" and name not in _FEATURE_INPUTS]
    if invalid:
        raise ValueError(
            f"Unknown MultiBranchCNN input(s): {invalid}. "
            f"Available: raw, {sorted(_FEATURE_INPUTS)}"
        )
    return inputs


def _raw_like_branch(out_channels: int = 64) -> nn.Sequential:
    return nn.Sequential(
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

        nn.Conv1d(128, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),

        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


def _track_branch(out_channels: int = 32) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=51, padding=25, bias=False),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.AvgPool1d(2),

        nn.Conv1d(16, out_channels, kernel_size=25, padding=12, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.AvgPool1d(2),

        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


class _MultiBranchCNN1D(nn.Module):
    """
    Multi-branch CNN. Each selected input gets its own 1-D branch; branch
    embeddings are concatenated and passed to a classifier head.

    Expects inputs dict keyed by the selected input names.
    """

    def __init__(
        self,
        input_names: list[str],
        n_classes: int = 4,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.input_names = input_names
        self.branches = nn.ModuleDict()
        total_dim = 0
        for name in input_names:
            if name == "raw" or name in {"autocorr", "zerocrossing"}:
                self.branches[name] = _raw_like_branch(out_channels=64)
                total_dim += 64
            else:
                self.branches[name] = _track_branch(out_channels=32)
                total_dim += 32

        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(total_dim, n_classes),
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        branch_outputs = []
        for name in self.input_names:
            x = inputs[name]
            if x.ndim == 2:
                x = x.unsqueeze(1)
            branch_outputs.append(self.branches[name](x))

        combined = torch.cat(branch_outputs, dim=1)
        return self.classifier(combined)


class MultiBranchCNNModel(TorchNNBase):
    required_inputs = ["raw", "pitchtrack"]

    @classmethod
    def required_inputs_for_options(cls, training_options: dict[str, any] | None = None) -> list[str]:
        training_options = training_options or {}
        return _parse_feature_inputs(
            training_options.get("feature_inputs", training_options.get("required_inputs"))
        )

    def __init__(self, training_options: dict[str, any]):
        TorchNNBase.__init__(self, training_options)
        self.required_inputs = self.required_inputs_for_options(training_options)
        self.build()

    def build(self) -> None:
        n_classes = int(self.training_options.get("n_classes", 4))
        p_drop    = float(self.training_options.get("p_drop", 0.1))
        self.model = _MultiBranchCNN1D(
            input_names=self.required_inputs,
            n_classes=n_classes,
            p_drop=p_drop,
        ).to(self.device)
