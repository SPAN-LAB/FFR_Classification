from abc import abstractmethod

from src.core import ffr_proc
from ...core import FFRPrep
from ...core import EEGSubject
from ...core import (
    EEGTrial,
)  # This isn't being used now but will be when ``infer`` is implemented
from .model_interface import ModelInterface

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
import json


class TorchNNBase(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        # Initialies subject and training_options attributes
        super().__init__(training_options)

        self.model: nn.Module | None = None
        self.device = None

        # Automatically attempt to use the GPU
        self.set_device()

    def set_device(self, use_gpu: bool = True):
        """
        If ``use_gpu`` is true, finds either a CUDA- or MPS-enabled GPU device. If none are found,
        or if ``use_gpu`` is false, uses the CPU instead.
        """
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

    def build(self):
        raise NotImplementedError("This method needs to be implemented")

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
        weight_decay = self.training_options.get("weight_decay", 0.1)
        min_impr = self.training_options.get("min_impr", 1e-3)

        prep = FFRPrep()

        folds = self.subject.folds
        total_correct = 0
        total_n = 0
        for i, fold in enumerate(folds):
            if verbose:
                print(f"\n===== Fold {i} =====")

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
                    batch_size = y_batch.size(0)
                    running_loss += loss.item() * batch_size
                    n_train += batch_size

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
                    print(
                        f"train loss={avg_train_loss:.4f}, val accuracy={val_acc:.4f}"
                    )

                if val_acc > best_val_acc + min_impr:
                    best_val_acc = val_acc
                    no_improve = 0
                    best_state = {
                        k: v.detach().cpu() for k, v in self.model.state_dict().items()
                    }
                else:
                    no_improve += 1
                    if no_improve >= 5:  # NOTE: using 5 in place of patience
                        break

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

                        self.subject.map_pred_to_trial(
                            index=int(trial_idx),
                            predicted_label=pred_label_1based,
                            prediction_distribution=dist,
                        )

        print("Theoretical dist:", self.subject.trials[32].prediction_distribution)
        return ffr_proc.get_accuracy(self.subject, True)

    def train(self):
        """
        To be implemented later
        """

    def infer(self, trials: list[EEGTrial]):
        """
        To be implemented later
        """
