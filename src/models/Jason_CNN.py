from __future__ import annotations

from typing import Any, Sequence
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedShuffleSplit

from ..core.eeg_subject import EEGSubject
from ..core.eeg_trial import EEGTrial
from ..core.ffr_prep import FFRPrep
from .model_interface import ModelInterface
from typing import Any


class Jason_CNN(ModelInterface):
    """
    TensorFlow CNN implementation that plugs into ModelInterface.

    Key assumptions (see bottom of file for more detail):
      - Each EEGTrial has a 1D array attribute with the FFR trace (default: 'ffr_dss').
      - Each EEGTrial has an integer class index attribute (default: 'mapped_label',
        expected to be in [0, n_classes-1]).
      - Each EEGTrial has `trial.prediction.prediction` as a writable field for
        storing predicted labels (your existing code seems to use this).
    """

    def __init__(
        self,
        training_options: dict[str, Any] | None = None,
        n_classes: int = 4,
        ds_factor: int = 8,
        epochs: int = 15,
        batch_size: int = 32,
        val_split: float = 0.10,
        group_size: int = 5,
        repeats: int = 6,
        bootstrap: bool = True,
        l2: float = 1e-5,
        sdrop: float = 0.10,
        head_drop: float = 0.30,
        lr: float = 1e-3,
        signal_attr: str = "ffr_dss",
        label_attr: str = "mapped_label",
    ):
        super().__init__(training_options or {})
        self.n_classes = n_classes
        self.ds_factor = ds_factor
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.group_size = group_size
        self.repeats = repeats
        self.bootstrap = bootstrap
        self.l2 = l2
        self.sdrop = sdrop
        self.head_drop = head_drop
        self.lr = lr
        self.signal_attr = signal_attr
        self.label_attr = label_attr
        self.model: tf.keras.Model | None = None

    def _build_model(self, input_len: int, n_classes: int) -> tf.keras.Model:
        inp = layers.Input(shape=(input_len, 1))
        x = layers.Conv1D(128, 9, padding="same", activation="relu")(inp)
        x = layers.SpatialDropout1D(self.sdrop)(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(128, 9, padding="same", activation="relu")(x)
        x = layers.SpatialDropout1D(self.sdrop)(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(256, 9, padding="same", activation="relu")(x)
        x = layers.SpatialDropout1D(self.sdrop)(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.head_drop)(x)

        out = layers.Dense(
            n_classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2),
        )(x)

        model = models.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _subaverage(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Your subaverage() copied into a method, using class hyperparams.
        X: (N, T, 1), y: (N,)
        """
        rng = np.random.default_rng()
        X_new, y_new = [], []

        for c in range(self.n_classes):
            idx = np.where(y == c)[0]
            if len(idx) < self.group_size:
                continue
            n_groups = max(1, (len(idx) // self.group_size) * self.repeats)
            for _ in range(n_groups):
                take = rng.choice(idx, size=self.group_size, replace=self.bootstrap)
                x_avg = X[take].mean(axis=0)
                X_new.append(x_avg)
                y_new.append(c)

        if not X_new:
            return None, None
        return np.stack(X_new, axis=0), np.array(y_new, dtype=np.int32)

    def _trials_to_xy(
        self,
        trials: Sequence[EEGTrial],
        require_labels: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Convert a list of EEGTrial instances into (X, y).

        X shape: (N, T, 1), y shape: (N,) or None if require_labels=False.
        """
        if not trials:
            raise ValueError("No trials provided.")

        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for tr in trials:
            # 1) Get the FFR signal
            sig = getattr(tr, self.signal_attr, None)
            if sig is None:
                raise AttributeError(
                    f"EEGTrial is missing '{self.signal_attr}'. "
                    "Either change Jason_CNN.signal_attr or add that attribute."
                )
            x = np.asarray(sig, dtype=np.float32)
            X_list.append(x)

            # 2) Get label if needed
            if require_labels:
                label = getattr(tr, self.label_attr, None)
                if label is None:
                    # Fallbacks in case your real label lives somewhere else
                    if hasattr(tr, "mapped_label") and tr.mapped_label is not None:
                        label = tr.mapped_label
                    elif hasattr(tr, "label") and tr.label is not None:
                        label = tr.label
                    elif hasattr(tr, "raw_label") and tr.raw_label is not None:
                        label = tr.raw_label
                    else:
                        raise AttributeError(
                            f"Could not find label on trial. Tried "
                            f"{self.label_attr}, mapped_label, label, raw_label."
                        )
                y_list.append(int(label))

        X = np.stack(X_list, axis=0)  # (N, T_raw)
        y = np.array(y_list, dtype=np.int32) if require_labels else None

        # Optional downsample (same as load_4tone_safe)
        if self.ds_factor and self.ds_factor > 1:
            X = X[:, :: self.ds_factor]

        # Per-trial z-score normalization + channel dimension
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True) + 1e-7
        X = ((X - m) / s)[..., np.newaxis]  # (N, T, 1)

        return X, y

    def _train_on_trials(self, train_trials: Sequence[EEGTrial]) -> None:
        """
        Train the CNN on a list of EEGTrial objects, using:
          - StratifiedShuffleSplit for train/val split
          - subaverage-based augmentation
          - class-balanced weights
          - early stopping & ReduceLROnPlateau
        """
        X, y = self._trials_to_xy(train_trials, require_labels=True)
        n_samples = X.shape[0]

        if n_samples < 2:
            raise RuntimeError("Not enough samples to train Jason_CNN.")

        # IMPORTANT: we assume labels are already in [0, n_classes-1]
        unique_labels = np.unique(y)
        if unique_labels.min() < 0 or unique_labels.max() >= self.n_classes:
            raise ValueError(
                f"Labels must be in [0, n_classes-1]. "
                f"Got labels {unique_labels.tolist()} with n_classes={self.n_classes}. "
                "Make sure EEGTrial.mapped_label is 0-based or adjust Jason_CNN."
            )

        # Train/val split (analogous to your StratifiedShuffleSplit)
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.val_split,
            random_state=42,
        )
        tr_idx, val_idx = next(sss.split(X, y))
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Subaverage augmentation (your subaverage())
        Xtr_sa, ytr_sa = self._subaverage(Xtr, ytr)

        Xtr_parts = [Xtr]
        ytr_parts = [ytr]
        if Xtr_sa is not None:
            Xtr_parts.append(Xtr_sa)
            ytr_parts.append(ytr_sa)

        Xtr_aug = np.concatenate(Xtr_parts, axis=0)
        ytr_aug = np.concatenate(ytr_parts, axis=0)

        # Build model (same architecture)
        input_len = X.shape[1]
        self.model = self._build_model(input_len, self.n_classes)

        # Bias init with log-priors
        class_counts = np.bincount(ytr_aug, minlength=self.n_classes)
        priors = class_counts.astype(np.float32)
        priors /= max(priors.sum(), 1.0)

        final_dense = self.model.layers[-1]
        try:
            final_dense.bias.assign(np.log(priors + 1e-8))
        except Exception:
            # If for some reason it fails (e.g. custom layer), just skip
            pass

        # Class weights (same idea as your script)
        class_weight = {
            i: float(len(ytr_aug) / (self.n_classes * max(class_counts[i], 1)))
            for i in range(self.n_classes)
        }

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
        rlr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

        self.model.fit(
            Xtr_aug,
            ytr_aug,
            validation_data=(Xval, yval),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[es, rlr],
            class_weight=class_weight,
        )

    def infer(self, trials: list[EEGTrial]):
        """
        Predict labels for a list of EEGTrial objects using the trained model.
        Also writes predictions into trial.prediction.prediction.
        """
        if not trials:
            return []

        if self.model is None:
            raise RuntimeError(
                "Jason_CNN.infer called before training. "
                "Call evaluate() or train() first."
            )

        X, _ = self._trials_to_xy(trials, require_labels=False)
        probs = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        preds = np.argmax(probs, axis=1)

        for trial, label in zip(trials, preds):
            trial.prediction.prediction = int(label)

        return preds.tolist()

    def train(self, output_path: str | None = None):
        if self.subject is None:
            raise RuntimeError(
                "No subject set. Call set_subject() before calling train()."
            )

        self._train_on_trials(self.subject.trials)

        if self.model is None:
            raise RuntimeError("Training failed; self.model is None.")

        if output_path is not None:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.model.save(output_path)


    def evaluate(self) -> float:
        """
        Cross-validated evaluation following ModelInterface docstring:

          1. Ensure folds exist on subject (via FFRPrep.make_folds).
          2. For each fold:
             - Train on trials not in the fold (using _train_on_trials).
             - Infer on the fold's trials.
             - Write predictions back into `trial.prediction.prediction`.
          3. Call `get_accuracy(subject)` (your existing function) and return it.
        """
        if self.subject is None:
            raise RuntimeError(
                "No subject set. Call set_subject() before calling evaluate()."
            )

        subject: EEGSubject = self.subject
        prep = FFRPrep()

        if not subject.folds:
            subject.folds = prep.make_folds(subject, num_folds=5)

        folds = subject.folds

        for fold in folds:
            test_trials = list(fold.trials)
            train_trials = [t for t in subject.trials if t not in test_trials]

            self._train_on_trials(train_trials)

            predicted_labels = self.infer(test_trials)

            for trial, label in zip(test_trials, predicted_labels):

                trial.prediction.prediction = int(label)

            tf.keras.backend.clear_session()
            self.model = None

        accuracy = get_accuracy(subject)
        return accuracy