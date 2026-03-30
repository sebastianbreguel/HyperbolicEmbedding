"""Hyperbolic Neural Network model (Ganea et al. 2018)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from layers import HypAct, HyperbolicMLR, HypLinear
from manifolds.base import Manifold


class HNN(nn.Module):
    """Two-layer Hyperbolic Neural Network classifier.

    Architecture (Ganea et al. 2018, Section 3.2):
        1. Map input from Euclidean space onto the ball: exp_0^c(x)
        2. HypLinear(in → hidden) + HypAct(ReLU in tangent space)
        3. HypLinear(hidden → out)
        4. Softmax: Euclidean nn.Softmax when c=0, HyperbolicMLR when c=1

    Args:
        manifold: Manifold instance (Euclidean or PoincareBall).
        in_features: Input feature dimension.
        out_features: Number of output classes.
        c: Curvature (0 for Euclidean, 1 for hyperbolic).
        hidden: Hidden layer dimension.
    """

    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
        c: float,
        hidden: int,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.linear1 = HypLinear(manifold, in_features, hidden, c, dropout=0)
        self.linear2 = HypLinear(manifold, hidden, out_features, c, dropout=0)
        self.hyp_act = HypAct(manifold, c, c, nn.ReLU())

        self.softmax: nn.Module = nn.Softmax(dim=1)
        if self.c == 1:
            self.softmax = HyperbolicMLR(manifold, out_features, out_features, c)

    def encode(self, x: Tensor) -> Tensor:
        """Return penultimate embedding (before classification head)."""
        x = self.manifold.proj(self.manifold.expmap0(x))
        x = self.hyp_act(self.linear1(x))
        return self.linear2(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.manifold.proj(self.manifold.expmap0(x))
        x = self.hyp_act(self.linear1(x))
        x = self.linear2(x)
        return self.softmax(x)
