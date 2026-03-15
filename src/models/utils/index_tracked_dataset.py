import torch
from torch.utils.data import Dataset

from ...core import EEGTrial

class IndexTrackedDataset(Dataset):
    def __init__(self, *, trials: list[EEGTrial]):
        self.trials = trials
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, index):
        
        trial = self.trials[index]
        data_tensor = torch.from_numpy(trial.data).float()
        label = torch.tensor(trial.enumerated_label).long()
        
        return {
            "x": data_tensor,
            "y": label, 
            "index": index,
            "trial_index": trial.trial_index
        }