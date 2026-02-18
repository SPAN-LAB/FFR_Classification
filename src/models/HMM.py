"""
Hidden Markov Model classifier for FFR sequence classification.

Design notes:
- One GaussianHMM per class, scored by log-likelihood (classic generative setup).
- Supports first and second temporal derivatives (delta / delta-delta).
- Adds practical robustness for small biomedical datasets:
  restarts, optional BIC/AIC state selection, class-prior scoring, and
  optional hybrid template term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import os
import pickle

import numpy as np
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_IMPORT_ERROR = None
except ImportError as e:
    GaussianHMM = Any  # type: ignore[assignment]
    _HMM_IMPORT_ERROR = e

from .utils import ModelInterface
from ..core.ffr_proc import get_accuracy


@dataclass
class _Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class HMM(ModelInterface):
    def __init__(self, training_options: dict[str, Any]):
        super().__init__(training_options or {})
        if _HMM_IMPORT_ERROR is not None:
            raise ImportError(
                "hmmlearn is required for HMM model. Install with: pip install hmmlearn"
            ) from _HMM_IMPORT_ERROR

        opts = self.training_options
        self.n_states = int(opts.get("n_states", 6))
        self.n_iter = int(opts.get("n_iter", 250))
        self.tol = float(opts.get("tol", 1e-3))
        self.covariance_type = str(opts.get("covariance_type", "diag"))
        self.min_covar = float(opts.get("min_covar", 1e-3))
        self.random_state = int(opts.get("random_state", 42))

        self.signal_attr = str(opts.get("signal_attr", "data"))
        self.label_attr = str(opts.get("label_attr", "label"))
        self.feature_mode = str(opts.get("feature_mode", "raw_delta_delta2"))
        self.normalize_features = bool(opts.get("normalize_features", True))
        self.per_sequence_zscore = bool(opts.get("per_sequence_zscore", True))
        self.temporal_downsample = max(1, int(opts.get("temporal_downsample", 1)))
        self.max_sequence_length = int(opts.get("max_sequence_length", 0))

        self.n_restarts = max(1, int(opts.get("n_restarts", 6)))
        self.max_fit_calls_per_class = max(1, int(opts.get("max_fit_calls_per_class", 12)))
        self.use_class_priors = bool(opts.get("use_class_priors", True))
        self.score_normalization = str(opts.get("score_normalization", "length"))

        self.auto_state_cap_by_samples = bool(opts.get("auto_state_cap_by_samples", True))
        self.min_sequences_per_state = max(1, int(opts.get("min_sequences_per_state", 3)))
        self.state_selection_criterion = str(opts.get("state_selection_criterion", "none")).lower()
        self.state_candidates = opts.get("state_candidates", None)

        self.hybrid_centroid_weight = float(opts.get("hybrid_centroid_weight", 0.0))
        self.default_eval_folds = int(opts.get("default_eval_folds", 5))
        self.verbose = bool(opts.get("verbose", False))

        self._class_models: dict[Any, GaussianHMM] = {}
        self._class_priors_log: dict[Any, float] = {}
        self._class_centroids: dict[Any, np.ndarray] = {}
        self._scaler: _Scaler | None = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[HMM] {msg}")

    def _trial_signal(self, trial) -> np.ndarray:
        x = getattr(trial, self.signal_attr, None)
        if x is None:
            raise AttributeError(f"Trial missing signal attr '{self.signal_attr}'.")
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < 3:
            raise ValueError("Sequence too short for HMM feature extraction.")
        return x

    def _trial_label(self, trial):
        if self.label_attr == "label":
            return trial.label
        y = getattr(trial, self.label_attr, None)
        if y is None:
            raise AttributeError(f"Trial missing label attr '{self.label_attr}'.")
        return y

    def _build_features(self, sig: np.ndarray) -> np.ndarray:
        if self.temporal_downsample > 1:
            sig = sig[:: self.temporal_downsample]
        if self.max_sequence_length > 0 and sig.shape[0] > self.max_sequence_length:
            stride = int(math.ceil(sig.shape[0] / self.max_sequence_length))
            sig = sig[::stride]

        raw = sig.reshape(-1, 1)
        if self.per_sequence_zscore:
            m = raw.mean(axis=0, keepdims=True)
            s = raw.std(axis=0, keepdims=True)
            raw = (raw - m) / np.maximum(s, 1e-8)

        if self.feature_mode == "raw":
            feat = raw
        elif self.feature_mode == "raw_delta":
            d1 = np.gradient(raw[:, 0]).reshape(-1, 1)
            feat = np.concatenate([raw, d1], axis=1)
        elif self.feature_mode == "raw_delta_delta2":
            d1 = np.gradient(raw[:, 0]).reshape(-1, 1)
            d2 = np.gradient(d1[:, 0]).reshape(-1, 1)
            feat = np.concatenate([raw, d1, d2], axis=1)
        else:
            raise ValueError(
                "Unsupported feature_mode. Use one of: "
                "'raw', 'raw_delta', 'raw_delta_delta2'."
            )
        return feat.astype(np.float64, copy=False)

    def _fit_scaler(self, seqs: list[np.ndarray]) -> _Scaler:
        x = np.concatenate(seqs, axis=0)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-8)
        return _Scaler(mean=mean, std=std)

    @staticmethod
    def _num_hmm_params(n_states: int, n_features: int, covariance_type: str) -> int:
        trans = n_states * (n_states - 1)
        start = n_states - 1
        means = n_states * n_features

        if covariance_type == "diag":
            covars = n_states * n_features
        elif covariance_type == "spherical":
            covars = n_states
        elif covariance_type == "full":
            covars = n_states * (n_features * (n_features + 1) // 2)
        elif covariance_type == "tied":
            covars = n_features * (n_features + 1) // 2
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

        return trans + start + means + covars

    def _candidate_states(self, n_sequences: int) -> list[int]:
        base = max(2, self.n_states)
        if isinstance(self.state_candidates, (list, tuple)) and len(self.state_candidates) > 0:
            candidates = sorted({max(2, int(s)) for s in self.state_candidates})
        elif self.state_selection_criterion in {"bic", "aic"}:
            candidates = [max(2, base - 2), max(2, base - 1), base, base + 1, base + 2]
            candidates = sorted(set(candidates))
        else:
            candidates = [base]

        if self.auto_state_cap_by_samples:
            cap = max(2, n_sequences // self.min_sequences_per_state)
            candidates = [s for s in candidates if s <= cap]
            if not candidates:
                candidates = [max(2, min(base, cap))]
        return candidates

    def _fit_one_class(self, class_seqs: list[np.ndarray], class_label: Any) -> GaussianHMM:
        n_sequences = len(class_seqs)
        if n_sequences < 2:
            raise ValueError(f"Class {class_label} has too few sequences ({n_sequences}).")

        states_candidates = self._candidate_states(n_sequences)
        x_cat = np.concatenate(class_seqs, axis=0)
        lengths = [len(s) for s in class_seqs]
        n_obs = x_cat.shape[0]
        n_feat = x_cat.shape[1]

        best_model: GaussianHMM | None = None
        best_objective = -np.inf
        n_candidates = len(states_candidates)
        budget = min(self.max_fit_calls_per_class, self.n_restarts * n_candidates)
        calls_per_candidate = max(1, budget // n_candidates)

        self._log(
            f"class={class_label} n_seq={n_sequences} n_obs={n_obs} "
            f"candidates={states_candidates} fit_budget={budget}"
        )

        for n_components in states_candidates:
            local_best_model: GaussianHMM | None = None
            local_best_ll = -np.inf

            for restart in range(calls_per_candidate):
                seed = self.random_state + 1009 * restart + 7919 * n_components
                model = GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    min_covar=self.min_covar,
                    random_state=seed,
                    verbose=False,
                    implementation="scaling",
                )
                try:
                    model.fit(x_cat, lengths)
                    ll = float(model.score(x_cat, lengths))
                    if np.isfinite(ll) and ll > local_best_ll:
                        local_best_ll = ll
                        local_best_model = model
                except Exception:
                    continue
            self._log(
                f"class={class_label} states={n_components} best_ll={local_best_ll:.3f}"
            )

            if local_best_model is None:
                continue

            if self.state_selection_criterion == "bic":
                p = self._num_hmm_params(n_components, n_feat, self.covariance_type)
                objective = -(-2.0 * local_best_ll + p * math.log(max(n_obs, 2)))
            elif self.state_selection_criterion == "aic":
                p = self._num_hmm_params(n_components, n_feat, self.covariance_type)
                objective = -(-2.0 * local_best_ll + 2.0 * p)
            else:
                objective = local_best_ll

            if objective > best_objective:
                best_objective = objective
                best_model = local_best_model

        if best_model is None:
            raise RuntimeError(f"Failed to fit class HMM for label {class_label}.")

        self._log(f"class={class_label} states={best_model.n_components} objective={best_objective:.3f}")
        return best_model

    def _fit(self, train_trials: list[Any]):
        labels = [self._trial_label(t) for t in train_trials]
        classes = sorted(set(labels), key=lambda x: str(x))
        if len(classes) < 2:
            raise ValueError("Need at least two classes to train HMM classifier.")

        feats = [self._build_features(self._trial_signal(t)) for t in train_trials]
        if self.normalize_features:
            self._scaler = self._fit_scaler(feats)
            feats = [self._scaler.transform(f) for f in feats]
        else:
            self._scaler = None

        by_class: dict[Any, list[np.ndarray]] = {c: [] for c in classes}
        by_class_waveforms: dict[Any, list[np.ndarray]] = {c: [] for c in classes}
        for trial, feat in zip(train_trials, feats):
            c = self._trial_label(trial)
            by_class[c].append(feat)
            by_class_waveforms[c].append(self._trial_signal(trial))

        self._class_models = {}
        for c in classes:
            self._log(f"Training class model for label={c} with {len(by_class[c])} sequences")
            self._class_models[c] = self._fit_one_class(by_class[c], c)

        if self.use_class_priors:
            counts = {c: len(by_class[c]) for c in classes}
            total = float(sum(counts.values()))
            k = len(classes)
            self._class_priors_log = {
                c: math.log((counts[c] + 1.0) / (total + k)) for c in classes
            }
        else:
            self._class_priors_log = {c: 0.0 for c in classes}

        if self.hybrid_centroid_weight > 0.0:
            self._class_centroids = {
                c: np.mean(np.stack(by_class_waveforms[c], axis=0), axis=0) for c in classes
            }
        else:
            self._class_centroids = {}

    def _score_trial(self, trial) -> dict[Any, float]:
        x = self._build_features(self._trial_signal(trial))
        if self._scaler is not None:
            x = self._scaler.transform(x)
        L = max(1, len(x))

        scores: dict[Any, float] = {}
        for c, model in self._class_models.items():
            try:
                s = float(model.score(x))
            except Exception:
                s = -np.inf

            if self.score_normalization == "length":
                s = s / L
            elif self.score_normalization == "none":
                pass
            else:
                raise ValueError("score_normalization must be 'none' or 'length'.")

            s += self._class_priors_log.get(c, 0.0)

            if self.hybrid_centroid_weight > 0.0 and c in self._class_centroids:
                centroid = self._class_centroids[c]
                sig = self._trial_signal(trial)
                mse = float(np.mean((sig - centroid) ** 2))
                template_sim = -mse
                w = min(max(self.hybrid_centroid_weight, 0.0), 1.0)
                s = (1.0 - w) * s + w * template_sim

            scores[c] = s
        return scores

    def infer(self, trials: list[Any]):
        if not self._class_models:
            raise RuntimeError("Model is not trained. Call evaluate() or train() first.")

        for trial in trials:
            scores = self._score_trial(trial)
            pred = max(scores, key=scores.get)
            trial.prediction = pred

            vals = np.array(list(scores.values()), dtype=np.float64)
            vals = vals - np.max(vals)
            probs = np.exp(vals)
            probs = probs / np.sum(probs)
            trial.prediction_distribution = {
                c: float(p) for c, p in zip(scores.keys(), probs.tolist())
            }

        return [t.prediction for t in trials]

    def evaluate(self) -> float:
        if self.subject is None:
            raise RuntimeError("No subject set. Call set_subject() first.")

        if not self.subject.folds:
            self.subject.fold(self.default_eval_folds)

        folds = self.subject.folds
        for i, test_trials in enumerate(folds):
            train_trials = [t for j, fold in enumerate(folds) if j != i for t in fold]
            self._log(f"Fold {i + 1}/{len(folds)}: train={len(train_trials)} test={len(test_trials)}")
            self._fit(train_trials)
            self.infer(test_trials)

        return float(get_accuracy(self.subject))

    def train(self, output_path: str | None = None):
        if self.subject is None:
            raise RuntimeError("No subject set. Call set_subject() first.")

        self._fit(self.subject.trials)

        if output_path is not None:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            payload = {
                "models": self._class_models,
                "priors": self._class_priors_log,
                "centroids": self._class_centroids,
                "scaler": self._scaler,
                "opts": self.training_options,
            }
            with open(output_path, "wb") as f:
                pickle.dump(payload, f)
