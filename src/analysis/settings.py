"""Canonical settings for classification analyses."""

from __future__ import annotations

from typing import Any


ANALYSIS_TYPES = ("subaverage", "data_amount")

GENERIC_MODEL_NAMES = (
    "BrainModule",
    "CNN",
    "EEGNet",
    "EEGNeX",
    "EEGNeXFFR",
    "FFNN",
    "FFNN_Cj",
    "RNN",
    "LSTM",
    "GRU",
    "LDA",
    "SVM",
)
GENERIC_NUM_EPOCHS = 50

TRIM_START_MS = 50
TRIM_END_MS = 250
NUM_FOLDS = 5

SUBAVERAGE_VALUES = (1, *range(5, 126, 5))
DATA_AMOUNT_SUBAVERAGE_SIZE = 5
DATA_AMOUNT_MIN = 100
DATA_AMOUNT_STRIDE = 100

RECURRENT_TRAINING_OPTIONS: dict[str, Any] = {
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "patience": 10,
    "hidden_size": 128,
    "frontend_channels": 32,
    "num_layers": 1,
    "p_drop": 0.2,
    "gradient_clip_norm": 1.0,
}

MODEL_TRAINING_OPTIONS: dict[str, dict[str, Any]] = {
    "CNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
    },
    "FFNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
    },
    "RNN": dict(RECURRENT_TRAINING_OPTIONS),
    "LSTM": dict(RECURRENT_TRAINING_OPTIONS),
    "GRU": dict(RECURRENT_TRAINING_OPTIONS),
    "MultiBranchCNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "feature_inputs": ["raw", "pitchtrack", "autocorr"],
    },
    "MultiBranchFFNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
    },
    "PitchCNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "patience": 50,
    },
    "EEGNeXFFR": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "patience": 50,
        "sinc_frontend": False,
    },
}

DEFAULT_TRAINING_OPTIONS: dict[str, Any] = {
    "num_epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-1,
    "patience": 50,
}


def training_options_for(model_name: str) -> dict[str, Any]:
    """Return a copy so individual runs cannot mutate global settings."""
    normalized_name = model_name.lower()
    for configured_name, options in MODEL_TRAINING_OPTIONS.items():
        if configured_name.lower() == normalized_name:
            return dict(options)
    return dict(DEFAULT_TRAINING_OPTIONS)


def generic_training_options_for(model_name: str) -> dict[str, Any]:
    """Return fixed-length training options for the supported generic models."""
    configured_name = next(
        (
            name
            for name in GENERIC_MODEL_NAMES
            if name.lower() == model_name.lower()
        ),
        None,
    )
    if configured_name is None:
        raise ValueError(
            f"Model {model_name!r} is not configured for generic analysis. "
            f"Expected one of: {', '.join(GENERIC_MODEL_NAMES)}."
        )

    options = training_options_for(configured_name)
    options["num_epochs"] = GENERIC_NUM_EPOCHS
    options["validation_ratio"] = 0.0
    return options
