"""Hyperbolic neural network layers (Ganea et al. 2018)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.modules.module import Module

import config
from manifolds.base import Manifold, ManifoldParameter


class HypLinear(nn.Module):
    """Hyperbolic linear layer via Möbius matrix-vector multiplication.

    Implements  y = M ⊗_c x  (and optionally adds a bias via Möbius addition)
    as described in Ganea et al. 2018, Section 3.2.

    Args:
        manifold: The Riemannian manifold (e.g. PoincareBall).
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        c: Curvature of the manifold.
        dropout: Dropout probability applied to the weight matrix.
    """

    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
        c: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = config.USE_BIAS
        self.bias = ManifoldParameter(
            torch.Tensor(out_features), requires_grad=True, c=c, manifold=manifold
        )
        self.weight = ManifoldParameter(
            torch.Tensor(out_features, in_features),
            requires_grad=True,
            c=c,
            manifold=manifold,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform init for weights, zero init for bias."""
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Möbius matvec and optional Möbius bias addition."""
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.proj(self.manifold.mobius_matvec(drop_weight, x))
        if self.use_bias:
            # Map bias to the ball, then add via Möbius addition.
            # Equivalent to exp_{res}(P_{0→res}(bias)) by the gyrovector identity.
            hyp_bias = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(self.bias.view(1, -1))))
            res = self.manifold.proj(self.manifold.mobius_add(res, hyp_bias))
        return res

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, c={self.c}"


class HypAct(Module):
    """Hyperbolic activation layer (Ganea et al. 2018, Section 3.2).

    Applies a Euclidean nonlinearity in the tangent space at the origin:
        HypAct(x) = exp_0^c( σ( log_0^c(x) ) )

    Args:
        manifold: The Riemannian manifold.
        c_in: Input curvature.
        c_out: Output curvature (currently must equal c_in).
        act: Euclidean activation function (e.g. nn.ReLU()).
    """

    def __init__(self, manifold: Manifold, c_in: float, c_out: float, act: nn.Module) -> None:
        super().__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        xt = self.act(self.manifold.logmap0(x))
        xt = self.manifold.proj_tan0(xt)
        return self.manifold.proj(self.manifold.expmap0(xt))

    def extra_repr(self) -> str:
        return f"c_in={self.c_in}, c_out={self.c_out}"
