

import torch
from torch import nn

class PairwiseContradictionScorer(nn.Module):
    """
    Binary contradiction scorer over pair encodings.
    Input: pooled embeddings (B, H)
    Output: sigmoid score in [0,1] for CONTRADICTION.
    """
    def __init__(self, hidden_size: int, mlp_hidden: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, pooled):  # pooled: (B, H)
        logits = self.classifier(pooled).squeeze(-1)
        return self.sigmoid(logits)  # (B,)