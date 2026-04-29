"""
SPAN Lab - FFR Classification

Filename: pitch_track.py
Description: Sliding-window pitch tracker adapted from autocorrtrack_fine.
"""

import numpy as np


def pitch_track(signal: np.ndarray, fs: float, minpitchlim: float = 70, maxpitchlim: float = 300) -> np.ndarray:
    """
    Sliding-window pitch tracker (adapted from autocorrtrack_fine).

    For each 40ms Hanning-windowed frame (1ms step), computes the normalized
    autocorrelation and finds the lag with the highest correlation in the
    [minpitchlim, maxpitchlim] Hz range.

    Parameters
    ----------
    signal : np.ndarray
        1-D raw EEG signal.
    fs : float
        Sampling frequency in Hz.
    minpitchlim : float
        Minimum pitch to detect (Hz).
    maxpitchlim : float
        Maximum pitch to detect (Hz).

    Returns
    -------
    np.ndarray
        1-D pitch track (Hz), length ≈ signal_duration_ms - 40.
    """
    windur     = 40  # ms
    winlen     = int(round((windur / 1000) * fs))
    envelope   = np.hanning(winlen)
    lagsamples = winlen

    data = np.atleast_1d(signal)
    if data.ndim > 1:
        data = data.flatten()

    corrpitch = []
    times     = []

    pos = 0
    while pos <= (len(data) - winlen):
        datacut  = data[pos : pos + winlen]
        datasamp = envelope * datacut

        r = np.correlate(datasamp, datasamp, mode='full')
        r = r[len(r) // 2:]   # positive lags only
        r = r[:lagsamples]

        if r[0] != 0:
            r = r / r[0]

        corrpitch.append(r)
        times.append(np.arange(lagsamples))

        pos = int(round(pos + 0.001 * fs))

    corrpitch = np.array(corrpitch).T   # (lagsamples, num_frames)
    times     = np.array(times).T

    # Convert lags → frequencies
    lags    = times[:, 0].astype(float)
    lags[0] = 1                          # avoid division by zero at lag 0
    freq    = fs / lags

    # Find lag indices corresponding to the pitch limits
    minpitchlimit_n = np.argmin(np.abs(freq - minpitchlim))
    maxpitchlimit_n = np.argmin(np.abs(freq - maxpitchlim))

    # Higher frequency = lower lag index, so ensure ordering is correct
    if maxpitchlimit_n > minpitchlimit_n:
        maxpitchlimit_n, minpitchlimit_n = minpitchlimit_n, maxpitchlimit_n

    corrpitchbuf = corrpitch[maxpitchlimit_n : minpitchlimit_n + 1, :]
    freqbuf      = freq[maxpitchlimit_n : minpitchlimit_n + 1]

    # For each frame, pick the frequency with the highest autocorrelation
    Ind        = np.argmax(corrpitchbuf, axis=0)
    pitchtrack = freqbuf[Ind]

    return pitchtrack.astype(np.float32)
