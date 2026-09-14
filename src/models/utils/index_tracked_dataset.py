import torch
import numpy as np
from torch.utils.data import Dataset

from ...core import EEGTrial


class IndexTrackedDataset(Dataset):
    def __init__(self, *, trials: list[EEGTrial], inputs: list[str] = ["raw"]):
        """
        Parameters
        ----------
        trials : list[EEGTrial]
        inputs : list[str]
            Which inputs to serve, matching the model's required_inputs.
            "raw" → trial.data (the raw waveform).
            Any other name → trial.features[name] (pre-computed by extract_features).
            Single input → batch["x"] tensor.
            Multiple inputs → batch["inputs"] dict {name: tensor} for multi-branch models.
        """
        self.trials = trials
        self.inputs = inputs

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):

        trial = self.trials[index]
        label = torch.tensor(trial.enumerated_label).long()

        result = {
            "y":           label,
            "index":       index,
            "trial_index": trial.trial_index,
        }

        if len(self.inputs) == 1:
            name = self.inputs[0]
            arr = trial.data if name == "raw" else trial.features[name]
            result["x"] = torch.from_numpy(np.asarray(arr)).float()
        else:
            tensors = {}
            for name in self.inputs:
                arr = trial.data if name == "raw" else trial.features[name]
                tensors[name] = torch.from_numpy(np.asarray(arr)).float()
            result["inputs"] = tensors

        return result
