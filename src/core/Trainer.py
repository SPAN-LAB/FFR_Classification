from abc import ABC, abstractmethod
from .EEGSubject import EEGSubjectInterface, EEGSubject
from typing import Any, Self, Callable

import importlib, json
from pathlib import Path
from typing import Dict, Optional

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
    def train(self, use_gpu: bool, num_epochs: int, lr: float, stopping_criteria: bool) -> Self: ...
        

    @abstractmethod
    def run(self, subject: EEGSubjectInterface) -> None: ...

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

    def trials_to_tensors(self, fold_idx: int,
                          as_torch: bool = True,
                          adjust_labels: bool = True):
        """
        Use a fold by index and return (X, y, indices). Ensures folds exist via stratified_folds.
        """

        fold = self.subject.folds[fold_idx]

        X = np.stack([trial.data for trial in fold], axis=0).astype(np.float32)
        y = np.asarray([int(trial.mapped_label) for trial in fold], dtype=np.int64)
        if adjust_labels:
            y = y - 1
        indices = np.asarray([trial.trial_index for trial in fold], dtype=np.int64)

        if not as_torch:
            return X, y, indices

        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(indices)

    def create_dataloaders(self, *, fold_idx: int, 
                           batch_size: int = 64,
                            val_size: float = 0.20,
                            test_size: float = 0.20,
                            add_channel_dim: bool = True,   # ONLY TRUE FOR CNNs
                            adjust_labels: bool = True,
                            shuffle_train: bool = True):

        X_np, y_np, indices_np = self.trials_to_tensors(fold_idx, as_torch=False, adjust_labels=adjust_labels)

        if add_channel_dim and X_np.ndim == 2:
            # Conv1d expects [N, C, T]
            X_np = X_np[:, None, :]

        num_samples = len(y_np)
        sample_indices = np.arange(num_samples)

        # 1) carve out test
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42 + fold_idx)
        trainval_indices, test_indices = next(sss_test.split(sample_indices, y_np))


        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=142 + fold_idx)
        rel_train_idx, rel_val_idx = next(sss_val.split(trainval_indices, y_np[trainval_indices]))
        train_indices = trainval_indices[rel_train_idx]
        val_indices   = trainval_indices[rel_val_idx]

        def dl_helper(idxs, *, shuffle: bool = False):
            X_t = torch.from_numpy(X_np[idxs])
            y_t = torch.from_numpy(y_np[idxs])
            idx_t = torch.from_numpy(indices_np[idxs])
            return DataLoader(TensorDataset(X_t, y_t, idx_t), batch_size=batch_size, shuffle=shuffle)

        train_dl = dl_helper(train_indices, shuffle=shuffle_train)
        val_dl   = dl_helper(val_indices,   shuffle=False)
        test_dl  = dl_helper(test_indices,  shuffle=False)
        return train_dl, val_dl, test_dl


    def train(self, use_gpu: bool, num_epochs: int, lr: float, stopping_criteria: bool):

        torch.manual_seed(42)
        np.random.seed(42)

        mod = importlib.import_module(f"src.models.{self.model_name}")
        ModelClass = getattr(mod, "Model")

        model_kwargs = {"input_size" : len(self.subject.trials[0].timestamps)}
        model: nn.Module = ModelClass(**model_kwargs)

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

        self.test_dataloaders : list[DataLoader] = []
        for fold_idx in range(len(folds)):
            fold_dir = subject_dir / f"fold{fold_idx + 1}"

            train_dl, val_dl, test_dl = self.create_dataloaders(
                fold_idx = fold_idx, add_channel_dim = add_channel_dim
            )
            self.test_dataloaders.append(test_dl)
            
            model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr = lr)

            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            best_val_acc = -1.0
            patience = 5
            min_impr = 1e-3
            no_improve = 0

            (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
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
                total_loss, total_correct, total_n = 0.0, 0, 0

                for batch in train_dl:
                    if len(batch) < 2:
                        raise ValueError("Each batch must be at least (x,y)")
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    preds = logits.argmax(dim=1)
                    total_loss += loss.item() * y.numel()
                    total_correct += (preds == y).sum().item()
                    total_n += y.numel()

                train_loss = total_loss / total_n
                train_acc = total_correct / total_n
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
            
                model.eval()
                with torch.no_grad():
                    v_loss, v_correct, v_n = 0.0, 0, 0
                    for xb, yb, _ in val_dl:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        logits = model(xb)
                        v_loss += criterion(logits, yb).item() * yb.numel()
                        v_correct += (logits.argmax(dim=1) == yb).sum().item()
                        v_n += yb.numel()

                    total_val_loss = v_loss / v_n
                    total_val_acc = v_correct / v_n
                    history["val_loss"].append(total_val_loss)
                    history["val_acc"].append(total_val_acc)

                    if total_val_acc > best_val_acc + min_impr:
                        best_val_acc = total_val_acc
                        no_improve = 0
                        torch.save(model.state_dict(), (fold_dir / "checkpoints" / "best.pt").as_posix())
                    else:
                        if stopping_criteria:
                            no_improve += 1

                print(f"[{epoch:03d}] train loss={train_loss:.4f} acc={train_acc:.3f} | "
                    f"val loss={total_val_loss:.4f} acc={total_val_acc:.3f}")

                if stopping_criteria and no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
                    break

            last_ckpt = None
            best_ckpt = None
            metrics_path = None

            last_path = fold_dir / "checkpoints" / "last.pt"
            torch.save(model.state_dict(), last_path.as_posix())
            last_ckpt = last_path.as_posix()
            bp = fold_dir / "checkpoints" / "best.pt"
            if bp.exists():
                best_ckpt = bp.as_posix()
            metrics_path = (fold_dir / "metrics.json").as_posix()
            with open(metrics_path, "w") as f:
                json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

        return {
            "history": history,
            "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
            "best_ckpt": best_ckpt,
            "last_ckpt": last_ckpt,
            "paths": {"metrics_json": metrics_path, "config_json": (fold_dir / "config.json").as_posix() if fold_dir else None},
        }
        
    def run(self, subject: EEGSubject):
        return None

        


