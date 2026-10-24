"""Frozen CNN-LSTM/waveform hybrid selected on 4T1006.

Independent copy of the archived experiment, discoverable as LSTM_hybrid.
Its settings select temporal trends and a 50/50 calibrated waveform blend.
See server_analysis/RNN_HYBRID_COMPARISON.md for the paired evaluation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .utils import TorchNNBase
from .EEGNeXFFR import _EEGNeXFFRNet, _LinearWithConstraint


class _LogEnergy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x.square())


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
        bandpass=None,
        bandpass_taps: int = 513,
        feature_channels: int = 8,
        clean_recurrent_input: bool = False,
        energy_frontend: bool = False,
        temporal_expansion_kernel: int = 16,
        head_pool_size: int = 1,
        input_slice: tuple[int, int] | None = None,
        forget_bias: float | None = None,
        trend_window: int = 1,
        include_deltas: bool = False,
    ):
        super().__init__()
        if (
            min(
                n_times,
                num_classes,
                hidden_size,
                pool_size,
                kernel_size,
                feature_channels,
            )
            < 1
        ):
            raise ValueError("Model dimensions must be positive")
        if recurrent_type not in {"rnn", "gru", "lstm"}:
            raise ValueError("recurrent_type must be rnn, gru, or lstm")
        self.input_slice = input_slice
        if input_slice is not None:
            start, stop = input_slice
            if not 0 <= start < stop <= n_times:
                raise ValueError("Invalid input time window")
            n_times = stop - start
        if bandpass is None:
            self.register_buffer("bandpass_filter", None)
        else:
            from scipy.signal import firwin

            if bandpass_taps < 3 or bandpass_taps % 2 == 0:
                raise ValueError("bandpass_taps must be odd and at least 3")
            kernel = firwin(bandpass_taps, bandpass, pass_zero=False, fs=16384)
            self.register_buffer(
                "bandpass_filter", torch.tensor(kernel, dtype=torch.float32)[None, None]
            )
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
            filter_1=feature_channels,
            kernel_block_3=temporal_expansion_kernel,
        )
        # Reuse the actual EEGNeXFFR blocks, removing only the final flatten.
        self.frontend = nn.Sequential(
            cnn.block_1,
            nn.Sequential(cnn.block_2, _LogEnergy())
            if energy_frontend
            else cnn.block_2,
            cnn.block_3,
            cnn.block_4,
            *list(cnn.block_5.children())[:-1],
        )
        self.clean_recurrent_input = clean_recurrent_input
        if clean_recurrent_input:
            # Apply the last dropout after the recurrent correction so the LSTM
            # receives a continuous feature sequence during training.
            self.frontend[-1] = nn.Identity()
        self.num_steps = cnn.final_layer.in_features // feature_channels
        if head_pool_size < 1 or head_pool_size > self.num_steps:
            raise ValueError("Invalid head_pool_size")
        self.head_pool = (
            nn.AvgPool1d(head_pool_size) if head_pool_size > 1 else nn.Identity()
        )
        recurrent_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[
            recurrent_type
        ]
        if trend_window < 1 or trend_window % 2 == 0:
            raise ValueError("trend_window must be a positive odd integer")
        self.trend_window = trend_window
        self.include_deltas = include_deltas
        recurrent_inputs = feature_channels * (2 if include_deltas else 1)
        self.recurrent = recurrent_class(
            recurrent_inputs, hidden_size, batch_first=True, bidirectional=False
        )
        if forget_bias is not None:
            if recurrent_type != "lstm":
                raise ValueError("forget_bias applies only to LSTM")
            with torch.no_grad():
                self.recurrent.bias_ih_l0[hidden_size : 2 * hidden_size].fill_(
                    forget_bias
                )
                self.recurrent.bias_hh_l0[hidden_size : 2 * hidden_size].zero_()
        self.projection = nn.Linear(hidden_size, feature_channels, bias=False)
        # Begin with the direct CNN features; learn a recurrent correction.
        nn.init.zeros_(self.projection.weight)
        self.dropout = nn.Dropout(dropout)
        self.classifier = _LinearWithConstraint(
            feature_channels * (self.num_steps // head_pool_size),
            num_classes,
            max_norm=0.25,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected single-channel waveforms")
        if self.input_slice is not None:
            x = x[..., self.input_slice[0] : self.input_slice[1]]
        if self.bandpass_filter is not None:
            padding = self.bandpass_filter.shape[-1] // 2
            x = F.conv1d(
                F.pad(x, (padding, padding), mode="reflect"), self.bandpass_filter
            )
        features = self.frontend(self.input_normalization(x).unsqueeze(2))
        features = features.squeeze(2).transpose(1, 2)  # [batch, time, channels]
        recurrent_features = features
        if self.trend_window > 1:
            recurrent_features = F.avg_pool1d(
                features.transpose(1, 2),
                self.trend_window,
                stride=1,
                padding=self.trend_window // 2,
                count_include_pad=False,
            ).transpose(1, 2)
        if self.include_deltas:
            delta = recurrent_features[:, 1:] - recurrent_features[:, :-1]
            delta = F.pad(delta, (0, 0, 1, 0)) * self.trend_window
            recurrent_features = torch.cat((recurrent_features, delta), dim=-1)
        sequence, _ = self.recurrent(recurrent_features)
        correction = self.projection(sequence)
        combined = (
            self.dropout(features + correction)
            if self.clean_recurrent_input
            else features + self.dropout(correction)
        )
        return self.classifier(self.head_pool(combined.transpose(1, 2)).flatten(1))


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
    def _core_train(self, **kwargs):
        self._waveform_model = None
        model = super()._core_train(**kwargs)
        options = self.training_options or {}
        blend = float(options.get("waveform_blend", 0.0))
        if not 0 <= blend <= 1:
            raise ValueError("waveform_blend must be between zero and one")
        if blend:
            import numpy as np
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import RidgeClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            trials = kwargs["trials"]
            features = self._waveform_features(trials)
            labels = np.array([trial.enumerated_label for trial in trials])
            estimator = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=float(options.get("waveform_alpha", 10000))),
            )
            # Calibration and scaling are fitted inside training-only inner folds.
            self._waveform_model = CalibratedClassifierCV(
                estimator, cv=3, method="sigmoid"
            )
            self._waveform_model.fit(features, labels)
        if not (self.training_options or {}).get("recalibrate_bn", False):
            return model
        from torch.utils.data import DataLoader
        from .utils.index_tracked_dataset import IndexTrackedDataset

        # Re-estimate only running statistics from training-fold inputs, with
        # dropout disabled. No test examples, labels, or optimizer steps are used.
        loader = DataLoader(
            IndexTrackedDataset(trials=kwargs["trials"]),
            batch_size=kwargs["batch_size"],
            shuffle=False,
        )
        modes = {module: module.training for module in model.modules()}
        norms = [
            module
            for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        momenta = {module: module.momentum for module in norms}
        model.eval()
        for module in norms:
            module.reset_running_stats()
            module.train()
        seen = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    inputs = self._extract_inputs(batch)
                    size = len(inputs)
                    for module in norms:
                        module.momentum = size / (seen + size)
                    model(inputs)
                    seen += size
        finally:
            for module, mode in modes.items():
                module.training = mode
            for module, momentum in momenta.items():
                module.momentum = momentum
        return model

    @staticmethod
    def _waveform_features(trials):
        import numpy as np
        from scipy.signal import butter, sosfiltfilt

        data = np.stack([trial.data for trial in trials])
        sos = butter(4, [80, 1000], btype="bandpass", fs=16384, output="sos")
        return sosfiltfilt(sos, data, axis=-1)[:, ::4].copy()

    def _core_infer(self, *, trials, batch_size):
        probabilities = super()._core_infer(trials=trials, batch_size=batch_size)
        blend = float((self.training_options or {}).get("waveform_blend", 0.0))
        if not blend or getattr(self, "_waveform_model", None) is None:
            return probabilities
        waveform_probs = self._waveform_model.predict_proba(
            self._waveform_features(trials)
        )
        for i, row in enumerate(waveform_probs):
            for label, probability in zip(self._waveform_model.classes_, row):
                probabilities[i][int(label)] = (1 - blend) * probabilities[i][
                    int(label)
                ] + blend * float(probability)
        return probabilities

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
            input_slice = None
            if options.get("crop_ms") is not None:
                import numpy as np

                if self.subject is None:
                    raise ValueError("crop_ms requires subject timestamps")
                start, stop = options["crop_ms"]
                timestamps = self.subject.trials[0].timestamps
                input_slice = (
                    int(np.searchsorted(timestamps, start, side="left")),
                    int(np.searchsorted(timestamps, stop, side="right")),
                )
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
                bandpass=options.get("bandpass"),
                bandpass_taps=int(options.get("bandpass_taps", 513)),
                feature_channels=int(options.get("feature_channels", 8)),
                clean_recurrent_input=bool(options.get("clean_recurrent_input", False)),
                energy_frontend=bool(options.get("energy_frontend", False)),
                temporal_expansion_kernel=int(
                    options.get("temporal_expansion_kernel", 16)
                ),
                head_pool_size=int(options.get("head_pool_size", 1)),
                input_slice=input_slice,
                forget_bias=options.get("forget_bias"),
                trend_window=int(options.get("trend_window", 1)),
                include_deltas=bool(options.get("include_deltas", False)),
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
