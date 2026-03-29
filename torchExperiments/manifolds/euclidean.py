"""Euclidean manifold — flat baseline for comparison with hyperbolic models."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Manifold


class Euclidean(Manifold):
    """Euclidean manifold with zero curvature.

    All manifold operations reduce to their standard Euclidean counterparts.
    Used as the baseline geometry when ``--model euclidean`` is selected.
    """

    def __init__(self, c: float | None = None) -> None:
        super().__init__()
        self.c = None  # zero curvature
        self.name = "Euclidean"

    def normalize(self, p: Tensor) -> Tensor:
        """Row-normalize p to unit norm in-place."""
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.0)
        return p

    def sqdist(self, p1: Tensor, p2: Tensor) -> Tensor:
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p: Tensor, dp: Tensor) -> Tensor:
        """Identity: Euclidean gradient is already the Riemannian gradient."""
        return dp

    def proj(self, p: Tensor) -> Tensor:
        """Identity: all points are already on the manifold."""
        return p

    def proj_tan(self, u: Tensor, p: Tensor) -> Tensor:
        return u

    def proj_tan0(self, u: Tensor) -> Tensor:
        return u

    def expmap(self, u: Tensor, p: Tensor) -> Tensor:
        """Geodesic step: p + u."""
        return p + u

    def logmap(self, p1: Tensor, p2: Tensor) -> Tensor:
        """Log map: p2 - p1."""
        return p2 - p1

    def expmap0(self, u: Tensor) -> Tensor:
        return u

    def logmap0(self, p: Tensor) -> Tensor:
        return p

    def mobius_add(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        """Vector addition."""
        return x + y

    def mobius_matvec(self, m: Tensor, x: Tensor) -> Tensor:
        """Standard matrix-vector multiplication."""
        return x @ m.transpose(-1, -2)

    def init_weights(self, w: Tensor, irange: float = 1e-5) -> Tensor:
        w.data.uniform_(-irange, irange)
        return w

    def ptransp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport is the identity in Euclidean space."""
        return v

    def ptransp0(self, x: Tensor, v: Tensor) -> Tensor:
        """Parallel transport from origin is the identity in Euclidean space."""
        return v

    def retr(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def retr_transp(self, x: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        y = self.retr(x, u)
        return y, self.ptransp(x, y, v)

    def component_inner(self, x: Tensor, u: Tensor, v: Tensor | None = None) -> Tensor:
        """Per-component inner product (equals u*v in Euclidean space)."""
        inner = u.pow(2) if v is None else u * v
        target_shape = torch.broadcast_shapes(x.shape, inner.shape)
        return inner.expand(target_shape)
