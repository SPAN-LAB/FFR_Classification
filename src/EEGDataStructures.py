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


class EEGTrial(EEGTrialProtocol):
    def __init__(
        self, 
        data: npt.NDArray[Any], 
        timestamps: npt.NDArray[Any], 
        trial_index: int,
        raw_label: Label,
        mapped_label: Label=None,
    ):
        self.data = data
        self.timestamps = timestamps
        self.trial_index = trial_index
        self.raw_label = raw_label
        self._mapped_label = mapped_label
    
    def trim(self, start_index: int, end_index: int):
        self.data = self.data[start_index: end_index + 1]
        self.timestamps = self.timestamps[start_index: end_index + 1]
    

class EEGSubject(EEGSubjectProtocol):
    def __init__(self, filepath: str):
        self.state = EEGSubjectStateTracker()
        self.trials: list[EEGTrialProtocol] = []
        self.source_filepath = filepath

        # Internal attribute for storing folds
        self._folds: list[list[EEGTrialProtocol]] | None = []

        self.load_data(filepath)
    
    def load_data(self, filepath: str):
        """
        A private helper method for getting the data using a given filepath.
        """
        # Get the raw data
        raw = read_mat(filepath)
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
        instructions = pd.read_csv(rule_filepath)
        labels_map = {}
        for column_label in instructions.columns:
            for raw_label in instructions[column_label]:
                labels_map[raw_label] = int(column_label)
        
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

    def grouped_trials(self) -> dict[Label, list[EEGTrialProtocol]]:
        """
        A private helper method for grouping trials in a dictionary according to their labels. i.e.
        all trials with label = 1 are in grouped_trials[1]... 
        """
        grouped_trials: dict[Label, list[EEGTrialProtocol]] = {}
        for trial in self.trials:
            if trial.mapped_label in grouped_trials:
                grouped_trials[trial.mapped_label].append(trial)
            else:
                grouped_trials[trial.mapped_label] = [trial]
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
            tr.map_labels()
    
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
        Use a fold by index from self.folds and return (X, y, indices).
        """
        if self.folds is None:
            raise ValueError("No folds have been created. Call stratified_folds(...) first.")
        if not (0 <= fold_idx < len(self.folds)):
            raise IndexError(f"fold_idx {fold_idx} out of range (0..{len(self.folds)-1}).")

        fold = self.folds[fold_idx]

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
              output_dir: Optional[str] = None,
              stopping_criteria: bool = False
    ):

        if self.folds is None or len(self.folds) == 0:
            self.stratified_folds(5) #CHANGE LATER

        torch.manual_seed(42)
        np.random.seed(42)

        add_channel_dim = True
        if model_name != "CNN":
            add_channel_dim = False

        num_folds = len(self.folds)
        for fold_idx in range(num_folds):
            train_dl, val_dl, _ = self.create_dataloaders(
                fold_idx=fold_idx, add_channel_dim = add_channel_dim
            )
            res = train_model(
                model_name=model_name,
                model_kwargs= {"input_size" : len(self.trials[0].timestamps)},
                train_loader=train_dl,
                val_loader=val_dl,
                num_epochs=num_epochs,
                lr=lr,
                output_dir=output_dir,
                device=self.device,
                stopping_criteria=stopping_criteria,
            )
            print(f"Fold {fold_idx+1} best val acc: {res['best_val_acc']:.3f}")

