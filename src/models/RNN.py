import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,  # kept for API compatibility; actual seq length is inferred at runtime
        n_classes: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        p_drop: float = 0.2,
        rnn_type: str = "gru",  # "gru" or "lstm"
    ) -> None:
        super().__init__()

        # Lightweight temporal feature extractor before RNN (helps learning on raw 1D traces)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        rnn_input_size = 128  # channels from feature_extractor
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=p_drop if num_layers > 1 else 0.0,
        )

        num_dirs = 2 if bidirectional else 1

        # Simple additive attention over time on top of RNN outputs
        self.attn_vector = nn.Linear(hidden_size * num_dirs, 1, bias=False)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * num_dirs),
            nn.Dropout(p_drop),
            nn.Linear(hidden_size * num_dirs, 128),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming for convs, Xavier for linears; leave RNN to default init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use 'relu' mode for Kaiming init; it's compatible with GELU layers
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts x of shape:
          - [N, T] (preferred from this codebase), or
          - [N, 1, T], or
          - [N, T, 1]
        Returns logits: [N, n_classes].
        """
        # Ensure channel-first for conv frontend: [N, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N, 1, T]
        elif x.dim() == 3:
            if x.shape[1] == 1 and x.shape[2] > 1:
                # Already [N, 1, T]
                pass
            elif x.shape[2] == 1 and x.shape[1] > 1:
                # [N, T, 1] -> [N, 1, T]
                x = x.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected 3D input shape for RNN model: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unexpected input shape for RNN model: {tuple(x.shape)}")

        # Feature extraction and prepare for RNN: [N, C, L] -> [N, L, C]
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)

        # RNN over sequence
        out, _ = self.rnn(x)  # out: [N, L, H*dirs]

        # Additive attention across time
        # scores: [N, L, 1] -> weights: [N, L, 1]
        scores = torch.tanh(out)
        scores = self.attn_vector(scores)
        attn = torch.softmax(scores, dim=1)
        pooled = (out * attn).sum(dim=1)  # [N, H*dirs]

        logits = self.head(pooled)
        return logits


