from .model_interface import NeuralNetworkInterface
from ..core.ffr_prep import FFRPrep
from ..core.eeg_subject import EEGSubject
from ..core.eeg_trial import EEGTrial
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class _CNN1D(nn.Module):
    """Expects input [B, 1, T]."""

    def __init__(self, n_classes: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNModel(NeuralNetworkInterface):
    """
    CNN that implements its own evaluate() using FFRPrep loaders,
    training+testing each fold in-memory (no checkpoint I/O).
    """

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        super().__init__(hyperparameters)

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

    def build(self) -> None:
        n_classes = int(self.hyperparameters.get("n_classes", 4))
        p_drop = float(self.hyperparameters.get("p_drop", 0.1))
        self.model = _CNN1D(n_classes=n_classes, p_drop=p_drop).to(self.device)

    def evaluate(self) -> float:
        if self.subject is None:
            raise RuntimeError("No subject set. Call set_subject(subject) first.")
        subject: EEGSubject = self.subject

        # Hyperparams
        epochs = int(self.hyperparameters.get("epochs", 15))
        lr = float(self.hyperparameters.get("lr", 1e-3))
        weight_decay = float(self.hyperparameters.get("weight_decay", 0.0))
        batch_size = int(self.hyperparameters.get("batch_size", 256))
        val_frac = float(self.hyperparameters.get("val_frac", 0.20))
        patience = int(self.hyperparameters.get("patience", 5))
        min_impr = float(self.hyperparameters.get("min_impr", 1e-3))
        adjust_labels = bool(self.hyperparameters.get("adjust_labels", True))

        torch.manual_seed(42)
        np.random.seed(42)

        # import FFRPrep class to use for folds and dataloaders
        prep = FFRPrep()

        folds = subject.folds or prep.make_folds(subject, num_folds=5)
        if hasattr(subject, "use_raw_labels"):
            subject.use_raw_labels()

        oof_true, oof_pred = [], []
        per_fold_best = []

        for fold_idx in range(len(folds)):
            # Fresh model per fold
            self.build()
            assert self.model is not None
            self.model.to(self.device)

            # Loaders via FFRPrep (Conv1d → add_channel_dim=True)
            train_dl, val_dl = prep.make_train_val_loaders(
                folds=folds,
                fold_idx=fold_idx,
                val_frac=val_frac,
                batch_size=batch_size,
                add_channel_dim=True,
                adjust_labels=adjust_labels,
            )
            test_dl = prep.make_test_loader(
                folds=folds,
                fold_idx=fold_idx,
                batch_size=batch_size,
                add_channel_dim=True,
                adjust_labels=adjust_labels,
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
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb, _ in test_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.model(xb).argmax(1)
                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            oof_true.append(np.asarray(y_true))
            oof_pred.append(np.asarray(y_pred))

        if not oof_true:
            return 0.0

        y_true_all = np.concatenate(oof_true, axis=0)
        y_pred_all = np.concatenate(oof_pred, axis=0)
        overall_acc = float((y_pred_all == y_true_all).mean())
        mean_best = float(np.mean(per_fold_best)) if per_fold_best else float("nan")

        subj_name = Path(self.subject.source_filepath).stem
        print(
            f"[{subj_name}] folds={len(folds)} | mean_best_val_acc={mean_best:.3f} | overall_acc={overall_acc:.3f}"
        )

        return overall_acc

    def train(self, output_path: str) -> None:
        return None  # NOTE: Placeholder, to be implemented later

    def infer(self, trials: list[EEGTrial]):
        return None  # NOTE: Placeholder, to be implemented later
