from abc import ABC, abstractmethod
from ..core.eeg_subject import EEGSubject
from ..core.eeg_trial import EEGTrial
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..core.ffr_prep import FFRPrep
from ..core.eeg_subject import EEGSubject
from ..core.eeg_trial import EEGTrial
from pathlib import Path
import importlib, json


class ModelInterface(ABC):
    def __init__(self):
        """
        Initializes a model with no subject bound to it.
        Call set_subject() after initializing.
        """
        self.subject: EEGSubject | None = (
            None  # NOTE: to be set by the set_subject() method
        )

    def set_subject(self, subject: EEGSubject):
        """
        Binds this model class to exactly one subject.
        Call this before calling any other methods in this class.
        """
        self.subject = subject

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluates the accuracy of the model using cross-validation in the following steps for the subject that was passed into this object
        1. The subject data is split into folds (see 'EEGSubject.Fold' and ffr_prep.make_folds())
        2. For `i` in 1 through n where n = number of folds:
            - Let the test set be (`EEGSubject.trials`) - (trials in fold `i`)
            - Let the train set be (`EEGSubject.trials`) - (trials NOT in fold `i`)
            - Train the model on the train set's `EEGTrial` instances
            - Use that model to create predictions on the test set's `EEGTrial` instances
        3. After training on all folds, 1 prediction is made for every `EEGTrial` of the subject
        4. We check each prediction against the actual label to obtain the accuracy of the model.
        This is the accuracy of the model trained on this `EEGSubject`

        :returns: Accuracy as a float for a model on one subject's data
        """
        ...

    @abstractmethod
    def train(self, output_path: str): ...

    @abstractmethod
    def infer(self, trials: list[EEGTrial]): ...


class NeuralNetworkInterface(ModelInterface, ABC):
    """
    Extends ModelInterface to add neural network specific behavior
    The following methods are what need to be defined in addition
    to the methods specified in ModelInterface
    """

    def __init__(self, hyperparameters: dict[str, Any]):
        """
        Adds hyperparameters to constructor
        hyperparameters: dictionary that defines the hyperparameters the model will use (such as num of epochs, stopping criteria, learning rate, etc...)
        """
        super().__init__()
        self.hyperparameters = hyperparameters
        self.model: Any | None = None  # NOTE: to be set by the build() method
        self.device: Any | None = None  # NOTE: to be set by the set_device() method

    @abstractmethod
    def set_device(self, use_gpu: bool = False):
        """
        Configure the compute device for this model.

        Args:
            use_gpu: If True, try a GPU (CUDA first, then Apple MPS if available);
                    otherwise force CPU.

        Behavior:
            - Must set `self.device` to a valid device.
            - If the model has already been built, move it to `self.device`.
            - All training/evaluation code must move input/label batches to
              `self.device` before forward passes.

        If a GPU is requested but none is available just use the CPU.
        """
        ...

    @abstractmethod
    def build(self) -> None:
        """
        Construct the architecture of your model and store it in self.model
        """
        ...


class TorchNNBase(NeuralNetworkInterface, ABC):
    def __init__(self, hyperparameters: dict[str, Any]):
        super().__init__(hyperparameters=hyperparameters)

    def set_device(self, use_gpu: bool = False):
        if use_gpu and torch.cuda.is_available():
            dev = torch.device("cuda")
        elif (
            use_gpu
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
        self.device = dev
        if self.model is not None:
            self.model.to(dev)

    def evaluate(self) -> float:
        if self.subject is not None:
            subject: EEGSubject = self.subject
        else:
            raise RuntimeError(
                "No subject set. Call set_subject() before calling evaluate"
            )

        # Hyperparams
        epochs = int(self.hyperparameters.get("epochs", 15))
        lr = float(self.hyperparameters.get("lr", 1e-3))
        weight_decay = float(self.hyperparameters.get("weight_decay", 0.0))
        batch_size = int(self.hyperparameters.get("batch_size", 256))
        val_frac = float(self.hyperparameters.get("val_frac", 0.20))
        patience = int(self.hyperparameters.get("patience", 5))
        min_impr = float(self.hyperparameters.get("min_impr", 1e-3))
        adjust_labels = bool(self.hyperparameters.get("adjust_labels", True))
        n_classes = int(self.hyperparameters.get("n_classes", 4))  # used for ROC

        torch.manual_seed(42)
        np.random.seed(42)

        prep = FFRPrep()
        folds = subject.folds or prep.make_folds(subject, num_folds=5)

        if subject.trials[0].mapped_label is None:
            subject.use_raw_labels()

        oof_true, oof_pred, oof_prob = [], [], []  # ← collect probs for ROC
        per_fold_best = []

        for fold_idx in range(len(folds)):
            # Fresh model per fold
            self.build()
            if self.model is not None:
                self.model.to(self.device)
            else:
                raise RuntimeError(
                    "self.model is None. Set model before calling evaluate()"
                )

            train_dl, val_dl = prep.make_train_val_loaders(
                folds=folds, fold_idx=fold_idx, val_frac=val_frac, batch_size=batch_size
            )
            test_dl = prep.make_test_loader(
                folds=folds,
                fold_idx=fold_idx,
                batch_size=batch_size,
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # Train with early stopping on val acc (keep best weights in memory)
            best_val_acc = -1.0
            best_state = None
            no_improve = 0

            for _ep in range(1, epochs + 1):
                self.model.train()
                for xb, yb, _ in train_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                # Validate
                self.model.eval()
                v_correct = v_n = 0
                with torch.no_grad():
                    for xb, yb, _ in val_dl:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        lg = self.model(xb)
                        v_correct += (lg.argmax(1) == yb).sum().item()
                        v_n += yb.numel()
                val_acc = (v_correct / max(v_n, 1)) if v_n else 0.0

                if val_acc > best_val_acc + min_impr:
                    best_val_acc = val_acc
                    no_improve = 0
                    best_state = {
                        k: v.detach().cpu() for k, v in self.model.state_dict().items()
                    }
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if best_state is not None:
                self.model.load_state_dict(best_state, strict=True)
                self.model.to(self.device)
            per_fold_best.append(float(best_val_acc))

            # Test on held-out fold
            self.model.eval()
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for xb, yb, _ in test_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    lg = self.model(xb)
                    probs = lg.softmax(dim=1)
                    preds = lg.argmax(1)
                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())  # (B, n_classes)

            oof_true.append(np.asarray(y_true))
            oof_pred.append(np.asarray(y_pred))
            oof_prob.append(np.asarray(y_prob))

        if not oof_true:
            return 0.0

        y_true_all = np.concatenate(oof_true, axis=0)
        y_pred_all = np.concatenate(oof_pred, axis=0)
        P = np.concatenate(oof_prob, axis=0)  # shape (N, n_classes)
        overall_acc = float((y_pred_all == y_true_all).mean())
        mean_best = float(np.mean(per_fold_best)) if per_fold_best else float("nan")

        subj_name = Path(subject.source_filepath).stem
        print(
            f"[{subj_name}] folds={len(folds)} | mean_best_val_acc={mean_best:.3f} | overall_acc={overall_acc:.3f}"
        )

        # cache for plotting
        self.last_eval = {
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "probs": P,
            "classes": list(range(n_classes)),
            "folds": len(folds),
            "mean_best_val_acc": mean_best,
            "overall_acc": overall_acc,
            "subject": subj_name,
            "device": str(self.device),
            "hyperparameters": dict(self.hyperparameters),
        }

        # quick JSON summary artifact
        out_dir = Path("outputs") / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{subj_name}_summary.json").open("w") as f:
            json.dump(
                {
                    "subject": subj_name,
                    "folds": len(folds),
                    "mean_best_val_acc": float(mean_best),
                    "overall_acc": float(overall_acc),
                    "device": str(self.device),
                },
                f,
                indent=2,
            )
        # NOTE: Need to add predicted labels to EEGTrial objects, use this to build CMs and ROCs
        return overall_acc
