"""Teacher-probe head (dataset-agnostic MLP on frozen features)."""

import torch.nn as nn


class Probe(nn.Module):
    """MLP probe; bottleneck = activation of the last hidden block (feeds the head)."""

    def __init__(self, in_dim, hidden_dims, n_classes, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        d = in_dim
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(
                nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)
            ))
            d = h
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, return_bottleneck=False):
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        if return_bottleneck:
            return logits, x
        return logits
