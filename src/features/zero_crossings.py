"""
SPAN Lab - FFR Classification

Filename: zero_crossings.py
Description: Zero-crossing feature. Returns a binary spike train — 1 where
             the signal crosses zero, 0 everywhere else. No amplitude info.
"""

import numpy as np


def zero_crossing_freq(signal: np.ndarray, fs: float) -> np.ndarray:
    out = np.zeros(len(signal), dtype=np.float32)
    signs = np.sign(signal)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs))[0]
    out[crossings] = 1.0
    return out
