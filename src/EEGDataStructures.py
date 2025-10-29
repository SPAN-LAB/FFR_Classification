from __future__ import annotations
from numpy import typing as npt
from typing import *
from enum import Enum, auto

import numpy as np
from random import shuffle
import pandas as pd 
from pymatreader import read_mat
import matplotlib.pyplot as plt
import seaborn as sns

from protocol import *
from train_model import train_model
from run_model import run_model
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import json

class EEGTrial(EEGTrialProtocol):
    def __init__(
        self, 
        data: npt.NDArray[Any], 
        timestamps: npt.NDArray[Any], 
        trial_index: int,
        raw_label: Label,
        mapped_label: Label | None=None,
        prediction: Label | None=None
    ):
        self.data = data
        self.timestamps = timestamps
        self.trial_index = trial_index
        self.raw_label = raw_label
        self._mapped_label = mapped_label
        self.prediction = prediction
    
    def trim(self, start_index: int, end_index: int):
        self.data = self.data[start_index: end_index + 1]
        self.timestamps = self.timestamps[start_index: end_index + 1]
    
    @property
    def mapped_label(self) -> Label:
        return self._mapped_label if self._mapped_label is not None else self.raw_label

    def map_label(self, label: Label):
        self._mapped_label = label
    

class EEGSubject(EEGSubjectProtocol):
    def __init__(self, filepath: str | None=None):
        self.state = EEGSubjectStateTracker()
        self.trials: list[EEGTrialProtocol] = []
        self.source_filepath = filepath

        # Internal attribute for storing folds
        self._folds: list[list[EEGTrialProtocol]] | None = []
        self.subject_name = None 
        self.model_used = None 

        if filepath:
            self.load_data(filepath)
    
    @staticmethod
    def pseudo_subject(subjects: list[EEGSubject]) -> EEGSubject:
        p_subject = EEGSubject()
        for subject in subjects:
            p_subject.trials += subject.trials
        return p_subject
    
    @staticmethod
    def visualize_grand_average(subjects: list[EEGSubject]):
        """
        Visualizes the grand average of EEG trials across subjects, grouped by raw label.
        Each subject's average for a label is plotted as a thin line, and the grand average
        across all subjects is plotted as a thick line.
        """
        raw_label_grouped_trials: dict[Label, list[list[EEGTrial]]] = {}
        for subject in subjects:
            grouped_trials = subject.grouped_trials(method="raw_label")
            for raw_label, trial_group in grouped_trials.items():
                if raw_label in raw_label_grouped_trials:
                    raw_label_grouped_trials[raw_label].append(trial_group)
                else:
                    raw_label_grouped_trials[raw_label] = [trial_group]
        
        # Set up the plot style
        sns.set_style("whitegrid")
        num_labels = len(raw_label_grouped_trials)
        fig, axes = plt.subplots(num_labels, 1, figsize=(12, 5 * num_labels))
        
        # Ensure axes is iterable even if there's only one subplot
        if num_labels == 1:
            axes = [axes]
        
        # Process each raw label
        for idx, (raw_label, list_of_grouped_trials) in enumerate(sorted(raw_label_grouped_trials.items())):
            ax = axes[idx]
            
            # Collect subject averages and timestamps
            subject_averages = []
            timestamps = None
            
            for trial_group in list_of_grouped_trials:
                if trial_group:  # Make sure the group is not empty
                    # Average across all trials for this subject
                    stacked_data = np.array([trial.data for trial in trial_group])
                    subject_avg = np.mean(stacked_data, axis=0)
                    subject_averages.append(subject_avg)
                    
                    # Get timestamps (should be the same for all trials)
                    if timestamps is None:
                        timestamps = trial_group[0].timestamps
            
            if not subject_averages:
                continue
            
            # Convert to numpy array for easier manipulation
            subject_averages = np.array(subject_averages)
            
            # Plot each subject's average as a thin grey line
            for i, subj_avg in enumerate(subject_averages):
                ax.plot(timestamps, subj_avg, alpha=0.5, linewidth=0.5, color='grey')
            
            # Compute and plot the grand average (average across subjects)
            grand_average = np.mean(subject_averages, axis=0)
            ax.plot(timestamps, grand_average, linewidth=3, color='black', 
                   label='Grand Average', zorder=10)
            
            # Styling
            ax.set_xlabel('Time (s)', fontsize=12, labelpad=10)
            ax.set_ylabel('Amplitude', fontsize=12, labelpad=10)
            ax.set_title(f'Grand Average for Raw Label: {raw_label}', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Add significant spacing between subplots and around the edges
        plt.tight_layout(pad=2.0, h_pad=5.0, w_pad=2.0)
        plt.show()
            
        # 

    
    def load_data(self, filepath: str):
        """
        A private helper method for getting the data using a given filepath.
        """
        # Get the raw data
        raw = read_mat(filepath)
        self.subject_name = Path(filepath).stem
        raw_data = raw["ffr_nodss"]
        timestamps = raw["time"]
        labels = raw["#subsystem#"]["MCOS"][3]

        # Create the EEGTrial instances
        self.trials = []
        for i, trial in enumerate(raw_data):
            self.trials.append(EEGTrial(
                data=trial,
                timestamps=timestamps,
                trial_index=i,
                raw_label=labels[i]
            ))
    
    def stratified_folds(self, num_folds: int=1) -> list[list[EEGTrialProtocol]]:
        # Skip recalculation if the data has already been split into folds
        if (self._folds is not None) and (len(self._folds) == num_folds):
            return self._folds
        
        folds: list[list[EEGTrialProtocol]] = [[] for _ in range(num_folds)]
        grouped_trials = self.grouped_trials()

        for _, trial_group in grouped_trials.items():
            shuffle(trial_group)
            for i, trial in enumerate(trial_group):
                folds[i % num_folds].append(trial)
            
            # We shuffle the folds to ensure each fold has approximately the same size. 
            # Notice that if we didn't shuffle, 23 trials divided into 5 folds would normally leave
            # group sizes of [5, 5, 5, 5, 3]. The aim with shuffling the folds 
            # is to flatten that distribution: [5, 4, 5, 5, 4]
            shuffle(folds)

        for fold in folds:
            shuffle(fold)
        
        self.state.mark_folded()

        self._folds = folds
        return folds

    def map_labels(self, rule_filepath: str):
        # Create a dictionary that maps from raw label to mapped label
        labels_map = {}
        with open(rule_filepath, 'r') as file:
            for line in file:
                values = line.strip().split(',')
                if values:  # Skip empty lines
                    mapped_label = int(values[0])  # First value is the mapped label
                    for raw_label in values[1:]:  # Remaining values are raw labels
                        if raw_label.strip():  # Skip empty values
                            labels_map[raw_label.strip()] = mapped_label
        
        # Set the mapped_label for each trial
        for trial in self.trials:
            trial.map_label(labels_map[trial.raw_label])
    
    def subaverage(self, size: int) -> EEGSubject: 
        grouped_trials = self.grouped_trials()
        
        subaveraged_trials: list[EEGTrialProtocol] = []
        for _, trial_group in grouped_trials.items():
            shuffle(trial_group)
            n = len(trial_group)
            for i in range(0, n // size, size):
                stacked_data = np.array([trial.data for trial in trial_group[i*size: (i+1)*size]])
                subaveraged_data = np.mean(stacked_data, axis=0)

                subaveraged_trials.append(EEGTrial(
                    data=subaveraged_data,
                    timestamps=trial_group[0].timestamps,
                    trial_index=len(subaveraged_trials),
                    raw_label=trial_group[0].mapped_label,
                    mapped_label=trial_group[0].mapped_label
                ))

        self.state.mark_subaveraged()

        self.trials = subaveraged_trials
        return self

    def grouped_trials(self, method: str="mapped_label") -> dict[Label, list[EEGTrialProtocol]]:
        """
        A private helper method for grouping trials in a dictionary according to their labels. i.e.
        all trials with label = 1 are in grouped_trials[1]... 
        """
        if method == "mapped_label":
            grouped_trials: dict[Label, list[EEGTrialProtocol]] = {}
            for trial in self.trials:
                if trial.mapped_label in grouped_trials:
                    grouped_trials[trial.mapped_label].append(trial)
                else:
                    grouped_trials[trial.mapped_label] = [trial]
            return grouped_trials
        else: # raw_label
            grouped_trials: dict[Label, list[EEGTrialProtocol]] = {}
            for trial in self.trials:
                if trial.raw_label in grouped_trials:
                    grouped_trials[trial.raw_label].append(trial)
                else:
                    grouped_trials[trial.raw_label] = [trial]
            return grouped_trials
    
    def trim(self, start_index: int, end_index: int):
        for trial in self.trials:
            trial.trim(start_index, end_index)

    def test_split(self, trials: list[EEGTrialProtocol], ratio: float) -> tuple[list[EEGTrialProtocol], list[EEGTrialProtocol]]:
        if ratio > 1 or ratio < 0:
            raise ValueError("Ratio must be in [0, 1].")

        shuffle(trials) # Could be removed if we are guaranteed already-shuffled trials 
        cutoff_index = int(len(trials) * ratio)
        return trials[:cutoff_index], trials[cutoff_index:]
    
    def use_raw_labels(self):
        for tr in self.trials:
            tr.map_label(tr.raw_label)
    
    def setDevice(self, use_gpu: bool = False):
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


    def to_arrays(self, as_torch: bool = False, adjust_labels: bool = True):
        """
        Converts the trials' data, labels, index into numpy arrays
        or torch tensors if `as_torch` is True.
        """
        X = np.stack([trial.data for trial in self.trials], axis = 0, dtype = np.float32)
        y = np.asarray([int(trial.mapped_label) for trial in self.trials], dtype = np.int64)
        # adjust labels:
        if adjust_labels:
            y = y - 1
        t = np.asarray(self.trials[0].timestamps, dtype = np.float32)
        indices = np.asarray([trial.trial_index for trial in self.trials], dtype=np.int64)

        if not as_torch:
            return X, y, t, indices
        
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y)
        t_torch = torch.from_numpy(t)
        indices_torch = torch.from_numpy(indices)

        return X_torch, y_torch, t_torch, indices_torch
    
    def trials_to_tensors(self, fold_idx: int,
                          as_torch: bool = True,
                          adjust_labels: bool = True):
        """
        Use a fold by index and return (X, y, indices). Ensures folds exist via stratified_folds.
        """
        folds = self.stratified_folds(len(self._folds) if self._folds else 5)
        if not (0 <= fold_idx < len(folds)):
            raise IndexError(f"fold_idx {fold_idx} out of range (0..{len(folds)-1}).")

        fold = folds[fold_idx]

        X = np.stack([trial.data for trial in fold], axis=0).astype(np.float32)
        y = np.asarray([int(trial.mapped_label) for trial in fold], dtype=np.int64)
        if adjust_labels:
            y = y - 1
        indices = np.asarray([trial.trial_index for trial in fold], dtype=np.int64)

        if not as_torch:
            return X, y, indices

        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(indices)

    
    def create_dataloaders(self, *, 
                           fold_idx: int, 
                           batch_size: int = 64,
                           train_size: float = 0.64, 
                           val_size: float = 0.16,
                           test_size: float = 0.20,
                           add_channel_dim: bool = True,   # ONLY TRUE FOR CNNs
                           adjust_labels: bool = True,
                           shuffle_train: bool = True
    ):
        
        X_np, y_np, indices_np = self.trials_to_tensors(fold_idx, as_torch=False, adjust_labels=adjust_labels)

        if add_channel_dim and X_np.ndim == 2:
            # For Conv1d: [N, 1, T]
            X_np = X_np[:, None, :]

        num_samples = len(y_np)
        sample_indices = np.arange(num_samples)

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42 + fold_idx)
        trainval_indices, test_indices = next(sss_test.split(sample_indices, y_np))

        # Now split train/val within the trainval set
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=142 + fold_idx)
        rel_train_idx, rel_val_idx = next(sss_val.split(trainval_indices, y_np[trainval_indices]))
        train_indices = trainval_indices[rel_train_idx]
        val_indices = trainval_indices[rel_val_idx]

        def dl_helper(idxs, *, shuffle: bool = False):
            X_t = torch.from_numpy(X_np[idxs])
            y_t = torch.from_numpy(y_np[idxs])
            idx_t = torch.from_numpy(indices_np[idxs])
            ds = TensorDataset(X_t, y_t, idx_t)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_dl = dl_helper(train_indices, shuffle=shuffle_train)
        val_dl   = dl_helper(val_indices,   shuffle=False)
        test_dl  = dl_helper(test_indices,  shuffle=False)
        return train_dl, val_dl, test_dl

    

    def train(self,*,model_name: str,
              num_epochs: int = 20,
              lr: float = 1e-3,
              stopping_criteria: bool = False
    ):
        self.model_used = model_name
        if not self._folds:
            self.stratified_folds(5)

        torch.manual_seed(42)
        np.random.seed(42)

        add_channel_dim = True
        if model_name != "CNN":
            add_channel_dim = False
        
        root = Path("outputs") / "train"
        subject_dir = root / self.subject_name

        folds = self._folds
        num_folds = len(folds)
        for fold_idx in range(num_folds):
            fold_dir = (subject_dir / f"fold{fold_idx+1}").as_posix()

            train_dl, val_dl, test_dl = self.create_dataloaders(
                fold_idx=fold_idx, add_channel_dim = add_channel_dim
            )
            self.test_dataloader = test_dl
            res = train_model(
                model_name=model_name,
                model_kwargs= {"input_size" : len(self.trials[0].timestamps)},
                train_loader=train_dl,
                val_loader=val_dl,
                num_epochs=num_epochs,
                lr=lr,
                output_dir=fold_dir,
                device=self.device,
                stopping_criteria=stopping_criteria,
            )
            print(f"[{self.subject_name}] Fold {fold_idx+1} best val acc: {res['best_val_acc']:.3f}")

    

    def test_model(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
        add_channel_dim = True
        if self.model_used != "CNN":
            add_channel_dim = False
        
        test_root = Path("outputs") / "test"
        subject_dir = test_root / self.subject_name
        subject_dir.mkdir(parents = True, exist_ok = True)

        train_root = Path("outputs") / "train" / self.subject_name



        num_folds = len(self._folds)
        results: list[dict] = []

        for fold_idx in range(num_folds):
            fold_dir = (subject_dir / f"fold{fold_idx+1}")
            fold_dir.mkdir(parents = True, exist_ok = True)

            #CHANGE LATER only works rn because random seeds are set
            #if seeds arent set test_dl might not be the same as the one created
            #when create_dataloaders is called in train_model
            _, _, test_dl = self.create_dataloaders(
                fold_idx=fold_idx, add_channel_dim = add_channel_dim
            )
            ckpt_dir = train_root / f"fold{fold_idx+1}" / "checkpoints"
            best = ckpt_dir / "best.pt"
            last = ckpt_dir / "last.pt"

            if best.exists():
                weights_path = best.as_posix()
                which = "best"
            elif last.exists():
                weights_path = last.as_posix()
                which = "last"
            else:
                raise FileNotFoundError(f"No checkpoint found for fold {fold_idx+1} in {ckpt_dir}")
            
            res = run_model(model_name = self.model_used,
                            model_kwargs = {"input_size" : len(self.trials[0].timestamps)},
                            dataloader= test_dl,
                            weights_path = weights_path,
                            device = self.device,
                            output_dir = fold_dir)

            res["fold"] = fold_idx + 1
            res["weights_used"] = weights_path
            res["which"] = which
            results.append(res)

            print(f"[{self.subject_name}] test fold {fold_idx+1}: acc={res['acc']:.3f} (n={res['n']}) → {res.get('saved_to','')} [{which}]")

        summary_path = subject_dir / "summary.json"
        mean_acc = (sum(r["acc"] for r in results) / len(results)) if results else 0.0
        with open(summary_path, "w") as f:
            json.dump(
                {"subject": self.subject_name, "model": self.model_used, "folds": results, "mean_acc": mean_acc},
                f,
                indent=2,
            )
        print(f"[{self.subject_name}] test summary → {summary_path.as_posix()} (mean_acc={mean_acc:.3f})")

        return results
       

    def summary(self) -> dict:
        num_trials = len(self.trials)
        labels = [t.mapped_label for t in self.trials]
        unique_labels = set(labels)
        num_classes = len(unique_labels)
        num_folds = len(self._folds) if self._folds else 0
        return {
            "num_trials": num_trials,
            "num_classes": num_classes,
            "num_folds": num_folds,
            "is_subaveraged": self.state.is_subaveraged(),
        }

