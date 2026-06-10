"""
SPAN Lab - FFR Classification

Feature registry. To add a new feature:
  1. Create src/features/<name>.py with a function (signal, fs) -> np.ndarray
  2. Import it here and add an entry to FEATURE_REGISTRY.
"""

import numpy as np
from .pitch_track import pitch_track
from .autocorr import autocorr

FEATURE_REGISTRY: dict[str, callable] = {
    "pitchtrack": pitch_track,
    "autocorr":   autocorr,
}


def compute_fs(timestamps: np.ndarray) -> float:
    """Derives sampling frequency (Hz) from a timestamps array (in ms)."""
    return 1.0 / float(np.mean(np.diff(timestamps))) * 1000
