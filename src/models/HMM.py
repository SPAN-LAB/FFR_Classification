"""One-HMM-per-class FFR classifier inspired by Llanos et al. (2017).

Pipeline per trial:
  1. Bandpass filter 80-1000 Hz
  2. Autocorrelation-based pitch tracking (40 ms frames, 1 ms hop)
  3. Train one Gaussian HMM per tone class (Baum-Welch via hmmlearn)
  4. Classify by maximum log-likelihood
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .utils.model_interface import ModelInterface
from ..core.eeg_trial import EEGTrial
from ..core.ffr_proc import get_accuracy


def _bandpass(sig: np.ndarray, fs: float, lo: float = 80.0, hi: float = 1000.0) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def _autocorr_pitch_track(sig: np.ndarray, fs: float,
                          windur: float = 40.0, hop_ms: float = 1.0,
                          minpitch: float = 70.0, maxpitch: float = 300.0
                          ) -> np.ndarray:
    """Autocorrelation-based pitch extraction.

    Returns (n_frames, 2) array with columns [pitchtrack, pitchstrength].
    """
    winlen = int(round((windur / 1000) * fs))
    hop = int(round((hop_ms / 1000) * fs))
    envelope = np.hanning(winlen)
    lagsamples = winlen

    data = np.atleast_1d(sig).ravel()

    pitchtrack = []
    pitchstrength = []

    # Precompute lag-to-frequency mapping and valid range
    lags = np.arange(lagsamples, dtype=np.float64)
    lags[0] = 1.0  # avoid division by zero
    freq = fs / lags
    minpitch_idx = np.argmin(np.abs(freq - minpitch))
    maxpitch_idx = np.argmin(np.abs(freq - maxpitch))
    if maxpitch_idx > minpitch_idx:
        maxpitch_idx, minpitch_idx = minpitch_idx, maxpitch_idx

    pos = 0
    while pos <= (len(data) - winlen):
        datacut = data[pos:pos + winlen]
        datasamp = envelope * datacut

        # Normalized autocorrelation
        r = np.correlate(datasamp, datasamp, mode='full')
        r = r[len(r) // 2:]
        r = r[:lagsamples]
        if r[0] != 0:
            r = r / r[0]

        # Find best peak in the pitch range
        r_range = r[maxpitch_idx:minpitch_idx + 1]
        freq_range = freq[maxpitch_idx:minpitch_idx + 1]

        best_idx = np.argmax(r_range)
        pitchtrack.append(freq_range[best_idx])
        pitchstrength.append(max(r_range[best_idx], 0.0))

        pos += hop

    return np.column_stack([pitchtrack, pitchstrength])


class HMM(ModelInterface):
    """HMM classifier for FFR tone classification."""

    def __init__(self, training_options: dict[str, Any] | None = None):
        super().__init__(training_options or {})
        opts = training_options or {}
        self.n_states = opts.get("n_states", 3)
        self.n_iter = opts.get("n_iter", 30)
        self.fs = opts.get("fs", 16384.0)
        self.seed = opts.get("seed", 42)

    def _extract(self, trials: list[EEGTrial]) -> list[np.ndarray]:
        out = []
        for t in trials:
            sig = np.asarray(t.data, dtype=np.float64).ravel()
            sig = _bandpass(sig, self.fs)
            out.append(_autocorr_pitch_track(sig, self.fs))
        return out

    def _fit_and_predict(
        self,
        train_trials: list[EEGTrial],
        test_trials: list[EEGTrial],
    ) -> np.ndarray:
        train_feats = self._extract(train_trials)
        test_feats = self._extract(test_trials)
        train_labels = np.array([t.raw_label for t in train_trials])

        # Standardize using training statistics
        scaler = StandardScaler()
        scaler.fit(np.concatenate(train_feats, axis=0))
        train_feats = [scaler.transform(f) for f in train_feats]
        test_feats = [scaler.transform(f) for f in test_feats]

        # One HMM per class
        classes = np.unique(train_labels)
        hmms: dict[Any, GaussianHMM] = {}

        for c in classes:
            idx = np.where(train_labels == c)[0]
            seqs = [train_feats[i] for i in idx]
            lengths = [s.shape[0] for s in seqs]
            X_cat = np.concatenate(seqs, axis=0)

            hmm = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=self.seed,
            )
            hmm.fit(X_cat, lengths)
            hmms[c] = hmm

        # Classify by max log-likelihood
        preds = []
        for feat in test_feats:
            best_class, best_score = None, -np.inf
            for c, hmm in hmms.items():
                score = hmm.score(feat)
                if score > best_score:
                    best_score = score
                    best_class = c
            preds.append(best_class)

        return np.array(preds)

    def evaluate(self) -> float:
        subject = self.subject
        folds = subject.folds

        for i, fold in enumerate(folds):
            test_trials = fold
            train_trials = [t for f_idx, f in enumerate(folds) for t in f if f_idx != i]
            preds = self._fit_and_predict(train_trials, test_trials)
            for trial, pred in zip(test_trials, preds):
                trial.prediction = pred

        return get_accuracy(self.subject)

    def train(self):
        pass

    def infer(self, trials: list[EEGTrial]):
        pass
