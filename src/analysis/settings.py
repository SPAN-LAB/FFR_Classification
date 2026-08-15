"""Canonical settings for classification analyses."""

from __future__ import annotations

from typing import Any


ANALYSIS_TYPES = ("subaverage", "data_amount")

TRIM_START_MS = 50
TRIM_END_MS = 250
NUM_FOLDS = 5

SUBAVERAGE_VALUES = (1, *range(5, 126, 5))
DATA_AMOUNT_SUBAVERAGE_SIZE = 5
DATA_AMOUNT_MIN = 100
DATA_AMOUNT_STRIDE = 100

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
