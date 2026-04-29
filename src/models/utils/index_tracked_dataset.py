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
            Single input → 1-D tensor. Multiple inputs → stacked (num_inputs, len) tensor.
        """
        self.trials = trials
        self.inputs = inputs

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):

        trial = self.trials[index]

        arrays = []
        for name in self.inputs:
            if name == "raw":
                arrays.append(trial.data)
            else:
                if name not in trial.features:
                    raise KeyError(
                        f"Feature '{name}' not found on trial. "
                        f"Did AnalysisPipeline.evaluate_model run extract_features first?"
                    )
                arrays.append(trial.features[name])

        if len(arrays) == 1:
            data = arrays[0]
        else:
            # Multi-input: stack as (num_inputs, feature_len)
            # Note: all arrays must have the same length for this to work.
            # For mixed lengths (raw + pitchtrack), a multi-branch CNN is needed.
            data = np.stack(arrays, axis=0)

        data_tensor = torch.from_numpy(np.asarray(data)).float()
        label       = torch.tensor(trial.enumerated_label).long()

        return {
            "x":           data_tensor,
            "y":           label,
            "index":       index,
            "trial_index": trial.trial_index,
        }
