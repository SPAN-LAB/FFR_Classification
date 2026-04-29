"""
SPAN Lab - FFR Classification

Filename: autocorr.py
Description: Full-signal autocorrelation feature.
"""

import numpy as np


def autocorr(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Full-signal autocorrelation. Output is the same length as the input signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D raw EEG signal.
    fs : float
        Sampling frequency (unused, kept for consistent feature function signature).

    Returns
    -------
    np.ndarray
        Normalized autocorrelation (lag-0 = 1), same length as signal.
    """
    r = np.correlate(signal, signal, mode='full')
    r = r[len(r) // 2:]   # positive lags only
    if r[0] != 0:
        r = r / r[0]      # normalize so lag-0 = 1
    return r.astype(np.float32)
