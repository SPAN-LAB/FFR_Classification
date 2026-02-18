from __future__ import annotations

from typing import Any, Sequence
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# from sklearn.model_selection import StratifiedShuffleSplit

from ..core.eeg_trial import EEGTrial
from .utils.model_interface import ModelInterface


class Jason_CNN(ModelInterface):
    """
    TensorFlow CNN implementation that plugs into ModelInterface.

    Key assumptions:
      - Each EEGTrial has a 1D array attribute with the signal (default: 'data').
      - Each EEGTrial has an integer class index attribute (default: 'mapped_label',
        expected to be in [0, n_classes-1]).
      - Each EEGTrial has `trial.prediction` as a writable field for storing
        predicted labels.
    """

    def __init__(
        self,
        training_options: dict[str, Any] | None = None,
        n_classes: int = 4,
        epochs: int = 15,
        batch_size: int = 32,
        val_split: float = 0.10,
        l2: float = 1e-5,
        sdrop: float = 0.10,
        head_drop: float = 0.30,
        lr: float = 1e-3,
        signal_attr: str = "data",
        label_attr: str = "mapped_label",
    ):
        super().__init__(training_options or {})
        opts = training_options or {}
        epochs = opts.get("num_epochs", epochs)
        batch_size = opts.get("batch_size", batch_size)
        lr = opts.get("learning_rate", lr)
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.l2 = l2
        self.sdrop = sdrop
        self.head_drop = head_drop
        self.lr = lr
        self.signal_attr = signal_attr
        self.label_attr = label_attr
        self.model: tf.keras.Model | None = None
        self._label_to_idx: dict[int, int] | None = None
        self._idx_to_label: dict[int, int] | None = None

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

        # Add channel dimension expected by Conv1D: (N, T) -> (N, T, 1)
        X = X[..., np.newaxis]

        return X, y

    def _train_on_trials(self, train_trials: Sequence[EEGTrial]) -> None:
        X, y_raw = self._trials_to_xy(train_trials, require_labels=True)
        n_samples = X.shape[0]
        print(
            f"[Jason_CNN] _train_on_trials: X shape={X.shape}, "
            f"n_samples={n_samples}, unique labels={np.unique(y_raw)}"
        )

        if n_samples < 2:
            raise RuntimeError("Not enough samples to train Jason_CNN.")

        # Build mapping from raw labels (e.g. 1,2,3,4) to 0..n_classes-1
        unique_labels = np.unique(y_raw)

        # Optional sanity check: number of distinct labels matches n_classes
        if len(unique_labels) != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} classes, "
                f"but found labels {unique_labels.tolist()} "
                f"(count={len(unique_labels)})."
            )

        # Create mapping dicts
        self._label_to_idx = {
            int(label): idx for idx, label in enumerate(sorted(unique_labels))
        }
        self._idx_to_label = {idx: label for label, idx in self._label_to_idx.items()}

        # Map raw labels to 0..n_classes-1
        y = np.array([self._label_to_idx[int(l)] for l in y_raw], dtype=np.int32)

        # Randomized split disabled per request.
        # sss = StratifiedShuffleSplit(
        #     n_splits=1,
        #     test_size=self.val_split,
        #     random_state=42,
        # )
        # tr_idx, val_idx = next(sss.split(X, y))
        # Xtr, ytr = X[tr_idx], y[tr_idx]
        # Xval, yval = X[val_idx], y[val_idx]
        n_val = max(1, int(len(X) * self.val_split))
        n_val = min(n_val, len(X) - 1)
        Xtr, ytr = X[:-n_val], y[:-n_val]
        Xval, yval = X[-n_val:], y[-n_val:]

        # Build model
        input_len = X.shape[1]
        self.model = self._build_model(input_len, self.n_classes)

        # Bias init with log-priors
        class_counts = np.bincount(ytr, minlength=self.n_classes)
        priors = class_counts.astype(np.float32)
        priors /= max(priors.sum(), 1.0)

        final_dense = self.model.layers[-1]
        try:
            final_dense.bias.assign(np.log(priors + 1e-8))
        except Exception:
            pass

        # Class weights
        class_weight = {
            i: float(len(ytr) / (self.n_classes * max(class_counts[i], 1)))
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
            Xtr,
            ytr,
            validation_data=(Xval, yval),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[es, rlr],
            class_weight=class_weight,
        )

    def infer(self, trials: list[EEGTrial]):
        """Predict labels for a list of EEGTrial objects and set trial.prediction."""
        if not trials:
            return []

        if self.model is None:
            raise RuntimeError(
                "Jason_CNN.infer called before training. "
                "Call evaluate() or train() first."
            )

        X, _ = self._trials_to_xy(trials, require_labels=False)
        probs = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        idx_preds = np.argmax(probs, axis=1)

        if self._idx_to_label is not None:
            label_preds = [int(self._idx_to_label[int(i)]) for i in idx_preds]
        else:
            # Fallback: assume indices are the labels
            label_preds = [int(i) for i in idx_preds]

        for trial, label in zip(trials, label_preds):
            trial.prediction = int(label)

        return label_preds

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

          1. Ensure folds exist on subject.
          2. For each fold:
             - Train on trials not in the fold (using _train_on_trials).
             - Infer on the fold's trials.
             - Infer writes predictions into `trial.prediction`.
        """
        if self.subject is None:
            raise RuntimeError(
                "No subject set. Call set_subject() before calling evaluate()."
            )

        folds = self.subject.folds
        if not folds:
            self.subject.fold(num_folds=5)
            folds = self.subject.folds

        for i, fold in enumerate(folds):
            print("doing a fold")
            test_trials = fold
            train_trials = []
            for j in range(len(folds)):
                if j == i:
                    continue
                for trial in folds[j]:
                    train_trials.append(trial)

            # Train on trials
            self._train_on_trials(train_trials)
            self.infer(test_trials)

            tf.keras.backend.clear_session()
            self.model = None

        t = 0
        s = 0

        for trial in self.subject.trials:
            t += 1
            if int(trial.raw_label) == trial.prediction:
                s += 1
        
        # accuracy = get_accuracy(subject)
        return s / t * 100
