"""
SPAN Lab - FFR Classification

Filename: spectrogram.py
Description: Python translation of stft_nike_sliding.m by Dr. Nike Gnanateja.
             Sliding window STFT spectrogram feature extractor.
"""

import numpy as np
from numpy import typing as npt
from scipy.signal.windows import hann


def spectrogram(signal: npt.ArrayLike, fs: float) -> npt.ArrayLike:
    """
    Computes a sliding window STFT spectrogram from a raw EEG signal.

    Parameters
    ----------
    signal : array-like
        Raw EEG signal (1D)
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    np.ndarray
        Flattened spectrogram of shape (n_timeframes * n_freqs,)
    """
    signal = np.array(signal, dtype=np.float64).flatten()
    fs     = int(round(fs))

    low_f    = 80
    high_f   = 300
    wind_dur = 40
    taper    = 5

    win_len  = round((wind_dur / 1000) * fs)
    ramp_len = round(fs * (wind_dur / 1000) * (2 * taper / 100))

    hann_win    = hann(ramp_len)
    plateau_len = win_len - ramp_len
    plateau     = np.ones(plateau_len)
    envelope    = np.concatenate([
        hann_win[:ramp_len // 2],
        plateau,
        hann_win[ramp_len // 2:]
    ])
    env_len = len(envelope)

    n_freq      = high_f - low_f
    hop_samples = round(0.001 * fs)
    pos         = 0
    cols        = []

    while pos <= len(signal) - win_len:
        chunk = signal[pos : pos + env_len]
        if len(chunk) < env_len:
            break

        tapered = envelope * chunk
        spec    = np.abs(np.fft.fft(tapered, n=fs))
        spec    = spec * 2.0 / len(spec)
        spec    = spec[low_f - 1 : high_f - 1]

        cols.append(spec)
        pos = round(pos + hop_samples)

    if len(cols) == 0:
        return np.zeros(n_freq, dtype=np.float32)

    return np.mean(np.stack(cols, axis=1), axis=1).astype(np.float32).flatten()