# src/models/cnn_simple.py
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Keras clone:
      Conv1D(128, k=9, same) + SpatialDropout1D(sdrop) + MaxPool1D(2)
      Conv1D(128, k=9, same) + SpatialDropout1D(sdrop) + MaxPool1D(2)
      Conv1D(256, k=9, same) + SpatialDropout1D(sdrop)
      GlobalAveragePooling1D + Dropout(head_drop)
      Dense(n_classes)  (CrossEntropyLoss handles softmax)
    Input:  [N, 1, T]
    Output: logits [N, n_classes]
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sdrop: float = 0.10,
        head_drop: float = 0.30,
    ):
        super().__init__()
        # "same" padding for k=9 -> pad=4
        pad = 9 // 2

        self.net = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=sdrop),  # SpatialDropout1D equivalent
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=sdrop),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=9, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=sdrop),
            nn.AdaptiveAvgPool1d(1),  # GlobalAveragePooling1D
        )
        self.head = nn.Sequential(
            nn.Flatten(),  # [N, 256, 1] -> [N, 256]
            nn.Dropout(p=head_drop),
            nn.Linear(256, num_classes),  # logits (no softmax)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, T]
        x = self.net(x)
        return self.head(x)
