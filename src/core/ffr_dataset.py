import torch
import numpy as np
from torch.utils.data import Dataset
from pymatreader import read_mat

class FFRDataset(Dataset):
    def __init__(self, file_path, use_clean_data=True):
        print(f"Reading v7.3 File: {file_path}")
        
        try:
            mat = read_mat(file_path)
        except Exception as e:
            raise RuntimeError(f"Read Error: {e}. (Did you install pymatreader?)")

        # 1. Select Data Source
        key_name = 'ffr_dss' if use_clean_data else 'ffr_nodss'
        if key_name not in mat:
            raise KeyError(f"Missing '{key_name}'. Keys found: {list(mat.keys())}")
            
        self.data = mat[key_name]
        raw_labels = mat['labels']
        
        # 2. Fix Dimensions (Auto-Transpose if sideways)
        # We need (Trials, Time)
        flat_labels = np.array(raw_labels).flatten()
        num_trials = len(flat_labels)
        
        if self.data.shape[0] != num_trials:
            print(f"Transposing data: {self.data.shape} -> ({self.data.shape[1]}, {self.data.shape[0]})")
            self.data = self.data.T

        # 3. Label Map (0, 1, 2, 3)
        self.unique_labels = np.unique(flat_labels)
        self.label_map = {val: i for i, val in enumerate(self.unique_labels)}
        self.int_labels = np.array([self.label_map[l] for l in flat_labels])

        print(f"Ready: {len(self.data)} trials. Classes: {self.unique_labels}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch expects (Channels, Time) -> (1, 4997)
        waveform = self.data[idx] 
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.int_labels[idx], dtype=torch.long)
        
        # TYPO FIXED HERE:
        return waveform, label
