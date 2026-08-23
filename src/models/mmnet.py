"""
MMNet: Monotonicity-constrained Multi-task Network.

Shared representation layers feed three independent binary output
heads (PPH, Sepsis, HIE). Monotonicity is enforced post-hoc via
weight clamping on a specified subset of "risk-increasing" input
features, so the model can never learn an inverted relationship
(e.g. "higher fever -> lower sepsis risk") for those features.
"""

import torch
import torch.nn as nn


class MMNet(nn.Module):
    def __init__(self, n_features: int, monotonic_feature_idx: list[int],
                 hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.monotonic_feature_idx = monotonic_feature_idx

        self.shared1 = nn.Linear(n_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.drop1 = nn.Dropout(0.3)

        self.shared2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.drop2 = nn.Dropout(0.3)

        self.pph_head = nn.Linear(hidden2, 1)
        self.sepsis_head = nn.Linear(hidden2, 1)
        self.hie_head = nn.Linear(hidden2, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.bn1(self.shared1(x)))
        h = self.drop1(h)
        h = self.relu(self.bn2(self.shared2(h)))
        h = self.drop2(h)

        pph = torch.sigmoid(self.pph_head(h))
        sepsis = torch.sigmoid(self.sepsis_head(h))
        hie = torch.sigmoid(self.hie_head(h))
        return torch.cat([pph, sepsis, hie], dim=1)

    @torch.no_grad()
    def clamp_monotonic_weights(self):
        """Call after every optimizer step. Clamps input-layer weight
        columns for monotonic-increasing features to be >= 0, so the
        first shared layer cannot assign a negative (risk-decreasing)
        weight to a clinically risk-increasing feature."""
        for idx in self.monotonic_feature_idx:
            self.shared1.weight[:, idx].clamp_(min=0)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())