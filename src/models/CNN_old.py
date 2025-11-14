import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, n_classes=4, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128,  kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),  nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 256,   kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(),    nn.MaxPool1d(2),

            nn.Conv1d(256, 128,  kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),    nn.MaxPool1d(2),

            nn.Conv1d(128, 64,  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.net(x)
    
    
