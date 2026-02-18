"""Discrete left-to-right HMM classifier with scalar vector quantization.

This module implements:
1) 1D vector quantization (k-means/LBG-style),
2) constrained discrete HMMs with Viterbi re-estimation,
3) a parallel multi-class wrapper (one HMM per class).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def _as_1d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input array must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values.")
    return arr


def _as_1d_int_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError("Sequence must be non-empty.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Sequence must contain integer symbols.")
    return arr.astype(int, copy=False)


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    out = a_max_safe + np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True))
    out = np.where(np.isfinite(a_max), out, -np.inf)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


class VectorQuantizer:
    """1D vector quantizer for scalar sequences."""

    def __init__(self) -> None:
        self._centroids: np.ndarray | None = None

    @property
    def centroids(self) -> np.ndarray:
        if self._centroids is None:
            raise ValueError("VectorQuantizer is not fitted.")
        return self._centroids.copy()

    def fit(
        self,
        x: np.ndarray,
        k: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 0,
    ) -> "VectorQuantizer":
        if k <= 0:
            raise ValueError("k must be > 0.")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        if tol < 0:
            raise ValueError("tol must be >= 0.")

        x1 = _as_1d_float_array(x)
        rng = np.random.default_rng(seed)

        replace = x1.size < k
        init_idx = rng.choice(x1.size, size=k, replace=replace)
        centroids = x1[init_idx].astype(float, copy=True)

        for _ in range(max_iter):
            dists = np.abs(x1[:, None] - centroids[None, :])
            labels = np.argmin(dists, axis=1)

            new_centroids = centroids.copy()
            for j in range(k):
                members = x1[labels == j]
                if members.size == 0:
                    new_centroids[j] = x1[rng.integers(0, x1.size)]
                else:
                    new_centroids[j] = np.mean(members)

            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if shift <= tol:
                break

        self._centroids = centroids
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            raise ValueError("VectorQuantizer is not fitted.")
        x1 = _as_1d_float_array(x)
        dists = np.abs(x1[:, None] - self._centroids[None, :])
        return np.argmin(dists, axis=1).astype(int, copy=False)


class DiscreteHMM:
    """Discrete HMM with constrained transition topology and hard-EM training."""

    def __init__(
        self,
        n_states: int,
        n_symbols: int,
        topology: str = "bakis3",
        smoothing: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if n_states <= 0 or n_symbols <= 0:
            raise ValueError("n_states and n_symbols must be > 0.")
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0.")
        if topology != "bakis3":
            raise ValueError("Only topology='bakis3' is supported.")
        if n_states != 3:
            raise ValueError("topology='bakis3' requires n_states=3.")

        self.n_states = n_states
        self.n_symbols = n_symbols
        self.topology = topology
        self.smoothing = smoothing
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._allowed = self._build_transition_mask()
        self.pi = np.zeros(self.n_states, dtype=float)
        self.pi[0] = 1.0

        self.A = self._init_transition_matrix()
        self.B = self._init_emission_matrix()

    def _build_transition_mask(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _normalize_rows(self, mat: np.ndarray) -> np.ndarray:
        row_sums = np.sum(mat, axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Cannot normalize matrix with non-positive row sum.")
        return mat / row_sums

    def _init_transition_matrix(self) -> np.ndarray:
        A = np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            allowed = np.where(self._allowed[i] > 0)[0]
            draw = self._rng.random(allowed.size) + self.smoothing
            draw /= np.sum(draw)
            A[i, allowed] = draw
        return A

    def _init_emission_matrix(self) -> np.ndarray:
        B = self._rng.random((self.n_states, self.n_symbols)) + self.smoothing
        return self._normalize_rows(B)

    def _check_symbol_sequence(self, seq: np.ndarray) -> np.ndarray:
        s = _as_1d_int_array(seq)
        if np.any((s < 0) | (s >= self.n_symbols)):
            raise ValueError(
                f"Symbol values must be in [0, {self.n_symbols - 1}], got out-of-range values."
            )
        return s

    def _log_A(self) -> np.ndarray:
        logA = np.full_like(self.A, -np.inf, dtype=float)
        nz = self.A > 0
        logA[nz] = np.log(self.A[nz])
        return logA

    def _log_B(self) -> np.ndarray:
        logB = np.full_like(self.B, -np.inf, dtype=float)
        nz = self.B > 0
        logB[nz] = np.log(self.B[nz])
        return logB

    def fit(self, sequences: List[np.ndarray], n_iter: int = 20) -> "DiscreteHMM":
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0.")
        if not sequences:
            raise ValueError("sequences must be non-empty.")

        checked = [self._check_symbol_sequence(seq) for seq in sequences]

        for _ in range(n_iter):
            A_counts = np.zeros_like(self.A)
            B_counts = np.zeros_like(self.B)

            for seq in checked:
                path = self.viterbi_path(seq)
                for t, sym in enumerate(seq):
                    B_counts[path[t], sym] += 1.0
                if len(seq) > 1:
                    for t in range(len(seq) - 1):
                        A_counts[path[t], path[t + 1]] += 1.0

            A_new = np.zeros_like(self.A)
            for i in range(self.n_states):
                allowed = self._allowed[i] > 0
                A_new[i, allowed] = A_counts[i, allowed] + self.smoothing
            self.A = self._normalize_rows(A_new)

            B_new = B_counts + self.smoothing
            self.B = self._normalize_rows(B_new)

        return self

    def viterbi_path(self, seq: np.ndarray) -> np.ndarray:
        s = self._check_symbol_sequence(seq)
        T = s.size
        S = self.n_states

        logA = self._log_A()
        logB = self._log_B()
        logpi = np.full(S, -np.inf, dtype=float)
        logpi[self.pi > 0] = np.log(self.pi[self.pi > 0])

        delta = np.full((T, S), -np.inf, dtype=float)
        psi = np.zeros((T, S), dtype=int)

        delta[0] = logpi + logB[:, s[0]]
        psi[0] = 0

        for t in range(1, T):
            for j in range(S):
                scores = delta[t - 1] + logA[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + logB[j, s[t]]

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def log_likelihood(self, seq: np.ndarray) -> float:
        s = self._check_symbol_sequence(seq)
        T = s.size
        S = self.n_states

        logA = self._log_A()
        logB = self._log_B()
        logpi = np.full(S, -np.inf, dtype=float)
        logpi[self.pi > 0] = np.log(self.pi[self.pi > 0])

        alpha = np.full((T, S), -np.inf, dtype=float)
        alpha[0] = logpi + logB[:, s[0]]

        for t in range(1, T):
            for j in range(S):
                alpha[t, j] = logB[j, s[t]] + _logsumexp(alpha[t - 1] + logA[:, j], axis=0)

        return float(_logsumexp(alpha[-1], axis=0))


class ParallelHMMClassifier:
    """Multi-class classifier using one discrete HMM per class."""

    def __init__(
        self,
        n_classes: int,
        n_states: int = 3,
        n_symbols: int = 50,
        smoothing: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if n_classes <= 1:
            raise ValueError("n_classes must be > 1.")
        if n_symbols <= 1:
            raise ValueError("n_symbols must be > 1.")
        self.n_classes = n_classes
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.smoothing = smoothing
        self.seed = seed

        self.vq = VectorQuantizer()
        self.hmms: List[DiscreteHMM] = []
        self._fitted = False

    def _check_raw_sequence(self, seq: np.ndarray) -> np.ndarray:
        return _as_1d_float_array(seq)

    def fit(
        self,
        X: List[np.ndarray],
        y: List[int],
        n_iter: int = 20,
    ) -> "ParallelHMMClassifier":
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0.")
        if len(X) == 0:
            raise ValueError("X must be non-empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        y_arr = np.asarray(y, dtype=int)
        if np.any((y_arr < 0) | (y_arr >= self.n_classes)):
            raise ValueError(f"y must contain class labels in [0, {self.n_classes - 1}].")

        X_checked = [self._check_raw_sequence(seq) for seq in X]
        pooled = np.concatenate(X_checked, axis=0)
        self.vq.fit(pooled, k=self.n_symbols, seed=self.seed)
        encoded = [self.vq.encode(seq) for seq in X_checked]

        self.hmms = []
        for c in range(self.n_classes):
            idx = np.where(y_arr == c)[0]
            if idx.size == 0:
                raise ValueError(f"No training sequences found for class {c}.")
            seqs_c = [encoded[i] for i in idx]
            hmm = DiscreteHMM(
                n_states=self.n_states,
                n_symbols=self.n_symbols,
                topology="bakis3",
                smoothing=self.smoothing,
                seed=self.seed + c,
            )
            hmm.fit(seqs_c, n_iter=n_iter)
            self.hmms.append(hmm)

        self._fitted = True
        return self

    def predict_logproba(self, X: List[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Classifier is not fitted.")
        if len(X) == 0:
            raise ValueError("X must be non-empty.")

        X_checked = [self._check_raw_sequence(seq) for seq in X]
        encoded = [self.vq.encode(seq) for seq in X_checked]

        out = np.zeros((len(encoded), self.n_classes), dtype=float)
        for i, seq in enumerate(encoded):
            for c in range(self.n_classes):
                out[i, c] = self.hmms[c].log_likelihood(seq)
        return out

    def predict(self, X: List[np.ndarray]) -> List[int]:
        scores = self.predict_logproba(X)
        return np.argmax(scores, axis=1).astype(int).tolist()


def _sample_left_to_right_states(T: int, rng: np.random.Generator) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be > 0.")
    A = np.array(
        [
            [0.75, 0.20, 0.05],
            [0.00, 0.85, 0.15],
            [0.00, 0.00, 1.00],
        ]
    )
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(3, p=A[states[t - 1]])
    return states


def _generate_synthetic_class(
    n_seq: int,
    T: int,
    means: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for _ in range(n_seq):
        z = _sample_left_to_right_states(T, rng)
        x = means[z] + rng.normal(0.0, noise_std, size=T)
        out.append(x.astype(float))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    T = 22

    X0_train = _generate_synthetic_class(30, T, means=np.array([-1.0, -0.3, 0.2]), noise_std=0.10, rng=rng)
    X1_train = _generate_synthetic_class(30, T, means=np.array([0.6, 1.1, 1.6]), noise_std=0.10, rng=rng)
    X_train = X0_train + X1_train
    y_train = [0] * len(X0_train) + [1] * len(X1_train)

    X0_test = _generate_synthetic_class(20, T, means=np.array([-1.0, -0.3, 0.2]), noise_std=0.10, rng=rng)
    X1_test = _generate_synthetic_class(20, T, means=np.array([0.6, 1.1, 1.6]), noise_std=0.10, rng=rng)
    X_test = X0_test + X1_test
    y_test = np.array([0] * len(X0_test) + [1] * len(X1_test))

    clf = ParallelHMMClassifier(n_classes=2, n_states=3, n_symbols=50, smoothing=1e-2, seed=7)
    clf.fit(X_train, y_train, n_iter=15)
    pred = np.array(clf.predict(X_test))
    acc = float(np.mean(pred == y_test))
    print(f"Synthetic self-test accuracy: {acc:.3f}")
