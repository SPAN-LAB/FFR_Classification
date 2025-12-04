from abc import abstractmethod

from ...printing import lprint, unlock
from ...printing import ulprint as print

from src.core import ffr_proc
from ...core import FFRPrep
from ...core.ffr_proc import get_accuracy
from ...core import EEGSubject
from ...core import EEGTrial  # This isn't being used now but will be when ``infer`` is implemented
from .model_interface import ModelInterface

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
import json


class TorchNNBase(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        # Initialize ``self.training_options``
        super().__init__(training_options)

        self.model: nn.Module | None = None
        self.device = None

        self.set_device() # Automatically attempt to use the GPU

    def set_device(self, use_gpu: bool = True):
        """
        Searches for a compatible GPU device if ``use_gpu`` is True.
        If one isn't found, or if ``use_gpu`` is False, uses the CPU instead.
        """
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Set to MPS")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def build(self):
        raise NotImplementedError("This method needs to be implemented")

    def debug_RNN(self, n_trials: int = 16, num_epochs: int = 200):
        """
        Debug helper: try to overfit a tiny subset of the data.

        Trains the current model on at most `n_trials` samples from the first
        train fold. If the model / pipeline are correct, the accuracy on this
        tiny set should approach 1.0.
        """
        if self.subject is None:
            raise RuntimeError("Subject must be set before calling debug_RNN().")

        # Build a fresh model
        self.build()
        if self.model is None:
            raise RuntimeError("self.model is None after build() in debug_RNN().")
        self.model.to(self.device)

        prep = FFRPrep()
        folds = self.subject.folds

        # Just use fold 0's training loader
        train_dl, _ = prep.make_train_val_loaders(
            folds=folds,
            fold_idx=0,
            batch_size=self.training_options["batch_size"],
        )

        try:
            x_batch, y_batch = next(iter(train_dl))
        except StopIteration:
            raise RuntimeError("Training DataLoader is empty in debug_RNN().")

        # Take only a small subset
        x_batch = x_batch[:n_trials]
        y_batch = y_batch[:n_trials]

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        print("[debug_RNN] x_batch shape:", x_batch.shape)
        print("[debug_RNN] y_batch shape:", y_batch.shape, y_batch.dtype)
        print("[debug_RNN] y_batch:", y_batch.tolist())
        print(
            "[debug_RNN] x_batch mean/std:", x_batch.mean().item(), x_batch.std().item()
        )

        criterion = nn.CrossEntropyLoss()
        lr = self.training_options["learning_rate"]
        weight_decay = self.training_options.get("weight_decay", 0.0)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        for ep in range(1, num_epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            logits = self.model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_batch).float().mean().item()

            print(f"[debug_RNN] epoch {ep:03d}: loss={loss.item():.4f}, acc={acc:.4f}")

            # If we can perfectly fit this tiny subset, pipeline is probably OK
            if acc == 1.0:
                break
        return acc

    def evaluate(self, verbose: bool = True) -> float:
        """
        Uses K-fold CV to train and test on EEGSubject data
        and returns overall accuracy as a float

        Preconditions:
            -self.subject != None
            -self.model != None
            -self.subject.folds != None
        """
        batch_size = self.training_options["batch_size"]
        learning_rate = self.training_options["learning_rate"]
        num_epochs = self.training_options["num_epochs"]
        weight_decay = self.training_options["weight_decay"]


        weight_decay = self.training_options.get("weight_decay", 0.1)
        early_stopping = self.training_options.get("early_stopping", False)
        min_impr = self.training_options.get("min_impr", 1e-3)

        prep = FFRPrep()

        folds = self.subject.folds
        total_correct = 0
        total_n = 0

        for i, fold in enumerate(folds):
            # if verbose:
            #     print(f"\n===== Fold {i + 1} =====")

            self.build()
            if self.model is not None:
                self.model.to(self.device)
            else:
                raise RuntimeError(
                    "Self.model is None. Set model before calling evaluate"
                )

            train_dl, val_dl = prep.make_train_val_loaders(
                folds=folds, fold_idx=i, batch_size=batch_size
            )
            test_dl = prep.make_test_loader(
                folds=folds, fold_idx=i, batch_size=batch_size
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            # Train with early stopping on val acc:
            best_val_acc = -1.0
            best_state = None
            no_improve = 0
            for ep in range(1, num_epochs + 1):

                self.model.train()

                running_loss = 0.0
                n_train = 0

                for (
                    x_batch,
                    y_batch,
                ) in train_dl:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(x_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    current_batch_size = y_batch.size(0)
                    running_loss += loss.item() * current_batch_size
                    n_train += current_batch_size

                avg_train_loss = running_loss / max(n_train, 1)
                self.model.eval()
                v_correct = v_n = 0
                with torch.no_grad():
                    for x_batch, y_batch in val_dl:
                        x_batch, y_batch = (
                            x_batch.to(self.device),
                            y_batch.to(self.device),
                        )
                        logits = self.model(x_batch)
                        v_correct += (logits.argmax(1) == y_batch).sum().item()
                        v_n += y_batch.numel()

                val_acc = (v_correct / max(v_n, 1)) if v_n else 0.0

                if verbose:
                    lprint(
                        f"Fold [{i + 1}/{len(folds)}], train loss={avg_train_loss:.4f}, val accuracy={val_acc:.4f}"
                    )

                if val_acc > best_val_acc + min_impr:
                    best_val_acc = val_acc
                    no_improve = 0
                    best_state = {
                        k: v.detach().cpu() for k, v in self.model.state_dict().items()
                    }
                elif early_stopping:
                    no_improve += 1
                    if no_improve >= 5:  # NOTE: using 5 in place of patience
                        if verbose:
                            print(f"Early stopping at epoch {ep}")
                        break

            unlock()

            if best_state is not None:
                self.model.load_state_dict(best_state, strict=True)
                self.model.to(self.device)

            # Test on held out fold:
            self.model.eval()
            with torch.no_grad():
                for x_batch, y_batch, idx in test_dl:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    logits = self.model(x_batch)
                    probs = logits.softmax(dim=1)
                    preds = logits.argmax(1)

                    total_correct += (preds == y_batch).sum().item()
                    total_n += y_batch.numel()

                    probs_np = probs.cpu().numpy()
                    preds_np = preds.cpu().numpy()
                    idx_np = idx.cpu().numpy()

                    for trial_idx, pred_label_0based, prob_vec in zip(
                        idx_np, preds_np, probs_np
                    ):
                        # convert 0–3 → 1–4
                        pred_label_1based = int(pred_label_0based) + 1

                        # optionally also make the distribution keys 1–4
                        dist = {cls + 1: float(p) for cls, p in enumerate(prob_vec)}

                        self.subject.trials[trial_idx].set_prediction(int(pred_label_0based))
                        self.subject.trials[trial_idx].prediction_distribution = dist

        return get_accuracy(self.subject, True)

    def train(self):
        """
        To be implemented later
        """

    def infer(self, trials: list[EEGTrial]):
        """
        To be implemented later
        """
