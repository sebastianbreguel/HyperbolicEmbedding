"""Euclidean neural network layers (baseline counterparts to hyperbolic layers)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import Module


class Linear(Module):
    """Euclidean linear layer with dropout and activation.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        dropout: Dropout probability applied after the linear transform.
        act: Activation function (e.g. nn.ReLU()).
        use_bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        act: nn.Module,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.linear(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.act(hidden)
