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
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from train_model import train_model

# NOTE: This implementation was made for single-channel data. 
class EEGTrial:
    """
    Represents data from a single trial: data, timestamps, raw_label, mapped_label, and trial index.
    """
    def __init__(self, data: npt.NDArray, timestamps: npt.NDArray, raw_label: int, trial_index: int):
        """
        :param data: A 1D numpy array containing the data.
        :param timestamps: The clock time corresponding to each datapoint in `data`
        :param raw_label: This `EEGTrial`'s initial (unprocessed) label
        :param mapped_label: This `EEGTrial`'s actual (mapped) label 
        :param trial_index: This `EEGTrial`'s trial index
        """
        self.data = self.formatted_data(data)
        self.timestamps = self.formatted_timestamps(timestamps)
        self.raw_label = raw_label
        self.mapped_label = None # Initially None
        self.trial_index = trial_index

    @property
    def label(self):
        if self.mapped_label is not None:
            return self.mapped_label
        else:
            return self.raw_label

    def trim(self, start_index: int, end_index: int):
        """
        Trims `data` and `timestamps`, keeping only the values included in the parameters' bounds. 
        """
        self.data = self.data[start_index: end_index + 1]

        # Trim the timestamps too for consistent lengths 
        self.timestamps = self.timestamps[start_index: end_index + 1]

    def trim_by_time(self, start_time: float = 0.0):
        keep = (time >= t0) & (time <= t1)


    def map_labels(self, filename: str=None, map: Dict[int, int]=None): 
        """
        Maps from raw labels to class labels. Accepts either a filename or a dictionary for mapping.
        """
        if not map and not filename:
            # Used if the raw_label IS the intended mapped_label
            self.mapped_label = self.raw_label
            return
        elif not map:
            # Used if we want to use the filename to map
            map = self.create_map_from_csv(filename)
        
        # By this point, map is non-None
        if self.raw_label not in map:
            raise KeyError("The raw label was not found in the map.")
        self.mapped_label = map[self.raw_label]
    
    @staticmethod
    def create_map_from_csv(path: str) -> Dict[int, int]:
        """
        Creates a dictionary that maps from raw labels to actual labels. This function is used by 
        `map_labels`.

        :param path: the path to the CSV file
        """
        instructions = pd.read_csv(path)
        labels_map = {}
        for column_label in instructions.columns:
            for raw_label in instructions[column_label]:
                labels_map[raw_label] = int(column_label)
        return labels_map
                
    @staticmethod
    def formatted_data(data: npt.NDArray) -> npt.NDArray:
        """
        Raises an error if `data` is not 1D, and returns it otherwise.
        """
        if data.ndim != 1:
            raise ValueError("Only 1D arrays are permitted.")
        return data
    
    @staticmethod
    def formatted_timestamps(timestamps: npt.NDArray) -> npt.NDArray:
        """
        Raises an error if `data` is not 1D, and returns it otherwise.
        """
        if timestamps.ndim != 1:
            raise ValueError("Only 1D arrays are permitted.")
        return timestamps

    def visualize(self, y_range: Tuple[float, float] | None = None):
        """
        Visualize this trial's data over time.

        :param y_range: Optional (min, max) y-limits for the plot.
        """
        # Determine the y-range if one wasn't provided
        if y_range is None:
            y_range = (float(np.min(self.data)), float(np.max(self.data)))

        # Prefer provided timestamps; else use index-based time
        if isinstance(self.timestamps, np.ndarray) and self.timestamps.size == self.data.size:
            x = self.timestamps
            x_label = "Time"
        else:
            x = np.arange(len(self.data))
            x_label = "Index"

        plt.figure(figsize=(12, 4))
        sns.lineplot(x=x, y=self.data)
        plt.title(f"FFR Trial for Tone {self.label}")
        plt.ylim(y_range)
        plt.xlabel(x_label)
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()


class EEGSubject: 
    class DataState(Enum):
        """
        This enum tracks the modification state of this instance. This also makes it easier to 
        determine whether when a method of this instance can or cannot run, because one just checks 
        the state to get information about the condition of the other attributes. (For example, 
        one cannot train with cross validation without first having split the data into folds.)
        """
        
        # The data has not been modified
        UNMODIFIED = auto() 
        
        # The data has been subaveraged. 
        # This comes with the guarantee that`self._subaveraged_trials` is not None.
        # Mutually exclusive with `UNMODIFIED`.
        SUBAVERAGED = auto() 

        # The data has been split into folds.
        # This comes with the guarantee that `self._folds` is not None.
        FOLDED = auto() 

    def is_unmodified(self):
        return self.DataState.UNMODIFIED in self.state
    
    def is_subaveraged(self):
        return self.DataState.SUBAVERAGED in self.state
    
    def is_folded(self):
        return self.DataState.FOLDED in self.state
    
    def remove_state(self, state: DataState):
        self.state.remove(state)

    def add_state(self, state: DataState):
        # Don't allow adding `UNMODIFIED` state.
        if state is self.DataState.UNMODIFIED:
            raise ValueError("Cannot add `UNMODIFIED` state to the subject.")
        
        if state is self.DataState.SUBAVERAGED:
            self.state.add(self.DataState.SUBAVERAGED)
            if self.is_unmodified():
                self.remove_state(self.DataState.UNMODIFIED)
        elif state is self.DataState.FOLDED:
            self.state.add(self.DataState.FOLDED)
            if self.is_unmodified():
                self.remove_state(self.DataState.UNMODIFIED)

    def __init__(self, trials: Sequence[EEGTrial], source_filepath: str=None):
        if not trials and not source_filepath:
            raise ValueError("trials and source filepath cannot both be None.")
        if not trials:
            # Open the file 
            file = read_mat(source_filepath)

            # Get the raw data
            raw_data = file["ffr_nodss"]
            timestamps = file["time"]
            labels = file["labels"]

            # Create the EEGTrial instances
            trials = []
            for i, trial in enumerate(raw_data):
                trials.append(EEGTrial(
                    data=raw_data[i],
                    timestamps=timestamps,
                    raw_label=labels[i],
                    trial_index=i
                ))

        self.state = set([self.DataState.UNMODIFIED])
        self._trials = trials
        self._subaveraged_trials = None
        self.folds = None
        self.source_filepath = source_filepath
        self.device = None #initially none

    @property
    def trials(self):
        """
        Logic for using either `_trials` or `_subaveraged_trials`
        """
        if self.DataState.SUBAVERAGED in self.state:
            # The data has been subaveraged
            return self._subaveraged_trials
        else: 
            # The data has not been subaveraged
            return self._trials

    # DEPRECATED: Use EEGSubject(source_filepath=...) instead
    @staticmethod
    def init_from_filepath(path: str) -> EEGSubject:
        # Open the file 
        file = read_mat(path)

        # Get the raw data
        raw_data = file["ffr_nodss"]
        raw_data = raw_data.T
        timestamps = file["time"]
        labels = file["labels"]

        # Create the EEGTrial instances
        trials = []
        for i, trial in enumerate(raw_data):
            trials.append(EEGTrial(
                data=raw_data[i],
                timestamps=timestamps,
                raw_label=labels[i],
                trial_index=i
            ))
        return EEGSubject(trials, source_filepath=path)
    
    def map_labels(self, filepath: str):
        labels_map = EEGTrial.create_map_from_csv(filepath)
        for trial in self.trials:
            trial.map_labels(map=labels_map)

    def trim(self, start_index, end_index):
        for trial in self.trials:
            trial.trim(start_index=start_index, end_index=end_index)

    def subaverage(self, size: int):
        """
        Averages data values over `size` `EEGTrial`s who have the same `mapped_label`

        :param size: The number of trials each subaveraging is composed from. 

        TODO: Let the key be customizable (for example, use `raw_label` instead of `mapped_label`)
        """
        print(f"Length before subaveraging: {len(self.trials)}")
        # Group trials by their labels 
        grouped_trials = self.grouped_trials()

        # Randomize the order of each list of homogeneous trials 
        for _, homogeneous_trials in grouped_trials.items():
            shuffle(homogeneous_trials)

        # Subaverage the trials 
        subaveraged_trials: List[EEGTrial] = []
        for label, homogeneous_trials in grouped_trials.items():
            n = len(homogeneous_trials)
            i = 0
            while i + size <= n:
                stacked_data = np.array([trial.data for trial in homogeneous_trials[i: i + size]])
                subaveraged_data = np.mean(stacked_data, axis=0)

                # Create a new EEGTrial object and assign it an artificial trial index
                # which equals the length of `subaveraged_trials`
                subaveraged_trial = EEGTrial(
                    data=subaveraged_data, 
                    timestamps=homogeneous_trials[0].timestamps, 
                    raw_label=label,
                    trial_index=len(subaveraged_trials)
                )
                subaveraged_trials.append(subaveraged_trial)

                i += size

        # Store the trials in `self.subaveraged_trials`
        self._subaveraged_trials = subaveraged_trials

        # Update the state
        self.add_state(self.DataState.SUBAVERAGED)

        print(f"Length after subaveraging: {len(self.trials)}")
            
    def test_split(self, trials: Sequence[EEGTrial], ratio: float):
        """
        Splits `trials` input into a train set and test set according to the specified `ratio`. 

        :param trials: the trials to be split into train and test sets
        :param ratio: the ratio of `trials` that become train sets 
        """
        if ratio > 1 or ratio < 0:
            raise ValueError("Ratio must be in [0, 1].")

        shuffle(trials) # Could be removed if we are guaranteed already-shuffled trials 
        cutoff_index = int(len(trials) * ratio)
        return trials[:cutoff_index], trials[cutoff_index:]

    def stratified_folds(self, num_folds):
        if self.is_folded():
            raise ValueError("Subject already folded. Cannot fold again.")
        
        folds = [[] for i in range(num_folds)]
        grouped_trials = self.grouped_trials()

        # For each label, we shuffle then distribute them over the folds
        for _, homogeneous_trials in grouped_trials.items():
            shuffle(homogeneous_trials)
            for i, trial in enumerate(homogeneous_trials):
                folds[i % num_folds].append(trial)
        
        # Shuffle the trials in each fold 
        for fold in folds:
            shuffle(fold)
        
        # Update the state
        self.add_state(self.DataState.FOLDED)

        # Store the folds
        self.folds = folds
    
    def visualize(self, label: int, y_range: Tuple[float, float] | None = None):
        """
        Visualize the subaveraged waveform over all trials for a given label (tone).

        :param label: Required. Visualize only trials with this label.
        :param y_range: Optional (min, max) y-limits applied to the plot.
        """
        groups = self.grouped_trials()
        if label not in groups:
            raise ValueError(f"No trials found for label {label}.")
        trials = list(groups[label])
        if not trials:
            raise ValueError("No trials available to visualize for the specified label.")

        # Stack and subaverage across all trials of the label
        stacked_data = np.stack([t.data for t in trials], axis=0)
        subavg = np.mean(stacked_data, axis=0)

        # Choose x-axis from timestamps if available; else sample index
        ref = trials[0]
        if isinstance(ref.timestamps, np.ndarray) and ref.timestamps.size == subavg.size:
            x = ref.timestamps
            x_label = "Time"
        else:
            x = np.arange(subavg.size)
            x_label = "Index"

        # Determine y-range if not provided
        if y_range is None:
            y_range = (float(np.min(subavg)), float(np.max(subavg)))

        plt.figure(figsize=(12, 4))
        sns.lineplot(x=x, y=subavg)
        plt.title(f"FFR Subaverage for Tone {label} (n={len(trials)})")
        plt.ylim(y_range)
        plt.xlabel(x_label)
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    
    def grouped_trials(self) -> Dict[int, Sequence[EEGTrial]]:
        """
        A private helper method for grouping trials in a dictionary according to their labels. i.e.
        all trials with label = 1 are in grouped_trials[1]... 
        """
        grouped_trials: Dict[int, Sequence[EEGTrial]] = {}
        for trial in self.trials:
            if trial.label in grouped_trials:
                grouped_trials[trial.label].append(trial)
            else:
                grouped_trials[trial.label] = [trial]
        return grouped_trials
    
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

