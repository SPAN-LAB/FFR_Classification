from abc import ABC, abstractmethod
from .EEGSubject import EEGSubjectInterface, EEGSubject
from .EEGTrial import EEGTrialInterface, EEGTrial
from typing import Any, Self, Callable

import importlib, json
from pathlib import Path
from typing import Dict, Optional

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class TrainerInterface(ABC):
    model_name: str
    subject: EEGSubjectInterface
    num_epochs: int
    lr: float
    stopping_criteria: bool
    subject_name: str

    device: torch.device


    @abstractmethod
    def train(self, use_gpu: bool, num_epochs: int, lr: float, stopping_criteria: bool) -> None: ...
        

    @abstractmethod
    def test(self) -> None: ...

class Trainer(TrainerInterface):
    def __init__(self, *, subject: EEGSubject, model_name: str):
        self.subject: EEGSubject = subject
        self.model_name: str = model_name

        self.subject_name = Path(subject.source_filepath).stem


    def set_device(self, use_gpu: bool = False):
        if use_gpu:
        # 1. Check for NVIDIA GPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")

            # 2. Check for Apple Silicon GPU
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")

            # 3. Fallback if GPU requested but none found
            else:
                self.device = torch.device("cpu")
        else:
            # 4. GPU not requested
            self.device = torch.device("cpu")

    def _trials_to_np(self, trials: list[EEGTrial], *, add_channel_dim: bool, adjust_labels: bool):
        X = np.stack([t.data for t in trials], axis=0).astype(np.float32)
        y = np.asarray([int(t.mapped_label) for t in trials], dtype=np.int64)
        
        if adjust_labels: y -= 1
        idx = np.asarray([t.trial_index for t in trials], dtype=np.int64)
        if add_channel_dim and X.ndim == 2:  # [N, T] -> [N, 1, T] for Conv1d
            X = X[:, None, :]
        return X, y, idx
    
    def create_train_val_dls(self, *, fold_idx,
                             val_frac: float = 0.20, batch_size: int = 64,
                             add_channel_dim: bool = False,
                             adjust_labels: bool = True,
                             shuffle_train: bool = True):
        
        folds = self.subject.folds
        test_fold_idx = fold_idx

        train_trials = []
        for f, fold in enumerate(folds):
            if f == test_fold_idx:
                continue
            train_trials.extend(fold) 
        X_tr_full, y_tr_full, idx_tr_full = self._trials_to_np(
            train_trials, add_channel_dim=add_channel_dim, adjust_labels=adjust_labels
        )

        
        n = len(y_tr_full)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42 + test_fold_idx)
        tr_idx, val_idx = next(sss.split(np.arange(n), y_tr_full))

        
        def make_dl(X, y, idxs, *, shuffle: bool):
            X_t = torch.from_numpy(X[idxs])
            y_t = torch.from_numpy(y[idxs])
            i_t = torch.from_numpy(idx_tr_full[idxs])
            return DataLoader(TensorDataset(X_t, y_t, i_t), batch_size=batch_size, shuffle=shuffle)

        train_dl = make_dl(X_tr_full, y_tr_full, tr_idx, shuffle=shuffle_train)
        val_dl   = make_dl(X_tr_full, y_tr_full, val_idx, shuffle=False)
        return train_dl, val_dl
    
    def create_test_dl(self, *, test_fold_idx: int, batch_size: int = 64,
                       add_channel_dim: bool = False, adjust_labels: bool = True):
        
        folds = self.subject.folds
        test_trials = folds[test_fold_idx]
        X_te, y_te, idx_te = self._trials_to_np(
            test_trials, add_channel_dim=add_channel_dim, adjust_labels=adjust_labels
        )

        ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te), torch.from_numpy(idx_te))
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


    def train(self, use_gpu: bool, num_epochs: int, lr: float, stopping_criteria: bool):

        torch.manual_seed(42)
        np.random.seed(42)

        
        self.set_device(use_gpu)
        print(f"Using device: {self.device}")

        add_channel_dim = self.model_name.lower().startswith("cnn")

        if self.subject.folds is None:
            self.subject.fold(num_folds = 5)   
        folds = self.subject.folds

        #CHANGE LATER TO CHECK CONDITION:
        self.subject.use_raw_labels()
        
        #CHANGE LATER TO ALLOW USER TO SET OUTPUT FOLDER
        root = Path("outputs") / "train"
        subject_dir = root / self.subject_name

        mod = importlib.import_module(f"src.models.{self.model_name}")
        ModelClass = getattr(mod, "Model")
        model_kwargs = {"input_size": len(self.subject.trials[0].timestamps)}
        
        for fold_idx in range(len(folds)):
            fold_dir = subject_dir / f"fold{fold_idx + 1}"

            train_dl, val_dl = self.create_train_val_dls(fold_idx = fold_idx,
                                                          add_channel_dim = add_channel_dim)
            
            model: nn.Module = ModelClass(**model_kwargs)
            model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr = lr)

            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            best_val_acc = -1.0
            patience = 5
            min_impr = 1e-3
            no_improve = 0

            (fold_dir / "checkpoints").mkdir(parents=True, exist_ok =True)
            with open(fold_dir / "config.json", "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "model_kwargs": model_kwargs,
                    "num_epochs": num_epochs,
                    "lr": lr,
                    "early_stopping": {"enabled": bool(stopping_criteria), "patience": patience, "min_impr": min_impr},
                    "selection": "best_val_acc"
                }, f, indent=2)
        

            for epoch in range(1, num_epochs + 1):
                model.train()
                total_loss = total_correct = total_n = 0

                for xb, yb, _ in train_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    preds = logits.argmax(1)
                    total_loss += loss.item() * yb.numel()
                    total_correct += (preds == yb).sum().item()
                    total_n += yb.numel()

                train_loss = total_loss / total_n
                train_acc  = total_correct / total_n
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)

                model.eval()
                with torch.no_grad():
                    v_loss = v_correct = v_n = 0
                    for xb, yb, _ in val_dl:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        lg = model(xb)
                        v_loss    += criterion(lg, yb).item() * yb.numel()
                        v_correct += (lg.argmax(1) == yb).sum().item()
                        v_n       += yb.numel()
                    val_loss = v_loss / v_n
                    val_acc  = v_correct / v_n
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(val_acc)

                    if val_acc > best_val_acc + min_impr:
                        best_val_acc = val_acc
                        no_improve = 0
                        torch.save(model.state_dict(), (fold_dir / "checkpoints" / "best.pt").as_posix())
                    else:
                        if stopping_criteria:
                            no_improve += 1

                print(f"[Fold {fold_idx+1}/{len(folds)}] "
                    f"[{epoch:03d}] train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                    f"val loss: {val_loss:.4f} acc: {val_acc:.3f}")

                if stopping_criteria and no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
                    break

            # Save last and metrics per fold (side effects only)
            torch.save(model.state_dict(), (fold_dir / "checkpoints" / "last.pt").as_posix())
            with open((fold_dir / "metrics.json").as_posix(), "w") as f:
                json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)
        
    def test(self):
        torch.manual_seed(42); np.random.seed(42)
        add_channel_dim = self.model_name.lower().startswith("cnn")

        test_root = Path("outputs") / "test"
        subject_dir = test_root / self.subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)

        train_root = Path("outputs") / "train" / self.subject_name
        num_folds = len(self.subject.folds)

        mod = importlib.import_module(f"src.models.{self.model_name}")
        ModelClass = getattr(mod, "Model")
        model_kwargs = {"input_size": len(self.subject.trials[0].timestamps)}

        # Collect OOF predictions across folds (per-subject)
        oof_true, oof_pred, oof_score = [], [], []
        total_acc = 0.0

        for fold_idx in range(num_folds):
            fold_dir = subject_dir / f"fold{fold_idx + 1}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            test_dl = self.create_test_dl(test_fold_idx=fold_idx, add_channel_dim=add_channel_dim)

            ckpt_dir = train_root / f"fold{fold_idx+1}" / "checkpoints"
            weights_path = (ckpt_dir / "best.pt")
            if not weights_path.exists():
                weights_path = (ckpt_dir / "last.pt")
            if not weights_path.exists():
                raise FileNotFoundError(f"No checkpoint for fold {fold_idx+1} in {ckpt_dir}")

            model: nn.Module = ModelClass(**model_kwargs)
            model.to(self.device)
            state = torch.load(weights_path.as_posix(), map_location=self.device)
            model.load_state_dict(state, strict=True)
            model.eval()

            y_true, y_pred, y_prob = [], [], []

            with torch.no_grad():
                for xb, yb, _ in test_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)

                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)

            # Save per-fold if you want (optional)
            cm = confusion_matrix(y_true, y_pred)
            acc = (y_pred == y_true).mean()
            total_acc += acc

            with open(fold_dir / "test_results.json", "w") as f:
                json.dump({"test_acc": float(acc), "confusion_matrix": cm.tolist()}, f, indent=2)

            # Append to OOF collections
            oof_true.append(y_true)
            oof_pred.append(y_pred)
            oof_score.append(y_prob)

            print(f"[{self.subject_name}] test fold {fold_idx+1}: acc = {acc:.3f} n = {y_true.size}")

        # ---- Per-subject (overall) metrics from OOF predictions ----
        oof_true = np.concatenate(oof_true, axis=0)
        oof_pred = np.concatenate(oof_pred, axis=0)
        oof_score = np.concatenate(oof_score, axis=0)

        mean_acc = total_acc / num_folds
        print(f"[{self.subject_name}] mean accuracy = {mean_acc:.3f}")

        # Confusion Matrix (overall)
        cm_overall = confusion_matrix(oof_true, oof_pred)
        plt.figure()
        plt.imshow(cm_overall, cmap="Blues")
        plt.title("Overall Confusion Matrix (Per Subject)")
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(cm_overall.shape[0]):
            for j in range(cm_overall.shape[1]):
                plt.text(j, i, cm_overall[i, j], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(subject_dir / "confusion_matrix_overall.png")
        plt.close()

        with open(subject_dir / "overall_metrics.json", "w") as f:
            json.dump({
                "mean_acc": float(mean_acc),
                "confusion_matrix_overall": cm_overall.tolist()
            }, f, indent=2)

        # ROC (multi-class OvR) — uses OOF scores
        classes = np.unique(oof_true)            # should be 0..C-1 if you used adjust_labels=True
        y_true_bin = label_binarize(oof_true, classes=classes)

        # Per-class ROC + AUC
        plt.figure()
        aucs = {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], oof_score[:, i])
            roc_auc = auc(fpr, tpr)
            aucs[int(cls)] = float(roc_auc)
            plt.plot(fpr, tpr, lw=2, label=f"class {cls} (AUC={roc_auc:.2f})")
        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), oof_score.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, lw=2, linestyle="--", label=f"micro (AUC={auc_micro:.2f})")

        plt.plot([0,1],[0,1],"k--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Overall ROC (Per Subject, OOF)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(subject_dir / "roc_overall.png")
        plt.close()

        with open(subject_dir / "roc_overall.json", "w") as f:
            json.dump({"per_class_auc": aucs, "micro_auc": float(auc_micro)}, f, indent=2)
            
