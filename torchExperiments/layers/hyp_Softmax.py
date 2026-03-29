"""Hyperbolic multinomial logistic regression (Ganea et al. 2018, Eq. 25).

Partially based on https://github.com/geoopt/geoopt (copyright Maxim Kochurov).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from manifolds import arsinh, artanh
from torch import Tensor


class HyperbolicMLR(nn.Module):
    """Hyperbolic softmax layer (multinomial logistic regression on the Poincaré ball).

    Replaces the standard Euclidean softmax when the model operates in
    hyperbolic space (c=1). Each class k is parameterised by:
      - p_k: a point on the ball (encoded as a Euclidean vector mapped with expmap0)
      - a_k: a tangent vector at p_k (direction of the decision hyperplane)

    The log-probability for class k is (Ganea 2018, Eq. 25):
        ( λ_{p_k}^c * ||a_k|| / sqrt(c) ) *
        arsinh( 2*sqrt(c) * <-p_k ⊕_c x, a_k> / ((1 - c*||-p_k⊕x||^2) * ||a_k||) )

    Args:
        manifold: The Poincaré ball manifold instance.
        ball_dim: Embedding dimensionality.
        n_classes: Number of output classes.
        c: Curvature of the Poincaré ball.
    """

    def __init__(self, manifold: object, ball_dim: int, n_classes: int, c: float) -> None:
        super().__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.manifold = manifold
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        c = torch.as_tensor(self.c).type_as(x)
        p_vals_poincare = self.manifold.expmap0(self.p_vals)
        # a_vals are tangent vectors at p_vals.  hyperbolic_softmax applies the
        # λ_{p_k}^c factor internally (Eq. 25).  Do NOT pre-scale by the
        # conformal factor — that would cancel λ and distort decision boundaries.
        return hyperbolic_softmax(x, self.a_vals, p_vals_poincare, c)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.a_vals, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.p_vals, a=np.sqrt(5))


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def tensor_dot(x: Tensor, y: Tensor) -> Tensor:
    """Batched dot product: returns tensor of shape (|x|, |y|)."""
    return torch.einsum("ij,kj->ik", x, y)


def mobius_addition_batch(x: Tensor, y: Tensor, c: Tensor) -> Tensor:
    """Batched Möbius addition (-y) ⊕_c x for all (x_i, y_j) pairs.

    Args:
        x: Shape (B, D) — batch of query points.
        y: Shape (C, D) — class representative points (will be negated).
        c: Scalar curvature tensor.

    Returns:
        Tensor of shape (B, C, D).
    """
    xy = tensor_dot(x, y)           # (B, C)
    x2 = x.pow(2).sum(-1, keepdim=True)   # (B, 1)
    y2 = y.pow(2).sum(-1, keepdim=True)   # (C, 1)
    num = (1 + 2 * c * xy + c * y2.permute(1, 0)).unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y   # (B, C, D)
    denom = 1 + 2 * c * xy + c**2 * x2 * y2.permute(1, 0)
    return num / (denom.unsqueeze(2) + 1e-5)


def hyperbolic_softmax(x: Tensor, a: Tensor, p: Tensor, c: Tensor) -> Tensor:
    """Compute class logits via hyperbolic MLR (Ganea et al. 2018, Eq. 25).

    Args:
        x: Input points, shape (B, D).
        a: Tangent vectors (directions), shape (C, D).
        p: Class representatives on the ball, shape (C, D).
        c: Scalar curvature tensor.

    Returns:
        Logits of shape (B, C).
    """
    # λ_{p_k}^c * ||a_k|| / sqrt(c)  — scalar per class
    lambda_pkc = 2 / (1 - c * p.pow(2).sum(dim=1).clamp_max((1.0 / c) - 1e-4))
    k = lambda_pkc * torch.norm(a, dim=1).clamp_min(1e-7) / torch.sqrt(c)

    mob_add = mobius_addition_batch(-p, x, c)   # (B, C, D) — note: -p ⊕_c x

    # numerator: 2*sqrt(c) * <mob_add, a_k>  shape (C, B)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * a.unsqueeze(1), dim=-1)

    # denominator: ||a_k|| * (1 - c * ||mob_add||^2)  — clamped to avoid /0
    denom = torch.norm(a, dim=1, keepdim=True).clamp_min(1e-7) * (
        1 - c * mob_add.pow(2).sum(dim=2)
    ).clamp_min(1e-7)

    logit = k.unsqueeze(1) * arsinh(num / denom)
    return logit.permute(1, 0)   # (B, C)
