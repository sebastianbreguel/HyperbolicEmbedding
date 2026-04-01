"""Poincaré ball model of hyperbolic space (Ganea et al. 2018)."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Manifold
from .math_utils import artanh, tanh


class PoincareBall(Manifold):
    """Poincaré ball manifold with curvature -c.

    Convention: the ball has radius 1/sqrt(c), i.e.
        B_c^n = { x in R^n : c * ||x||^2 < 1 }

    All formulas follow Ganea et al. 2018 "Hyperbolic Neural Networks"
    (https://arxiv.org/abs/1805.09112).
    """

    def __init__(self, c: float) -> None:
        super().__init__()
        self.name = "PoincareBall"
        self.min_norm: float = 1e-7
        self.c = c
        # dtype-dependent numerical eps for boundary projection
        self.eps: dict[torch.dtype, float] = {torch.float32: 4e-3, torch.float64: 1e-5}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lambda_x(self, x: Tensor) -> Tensor:
        """Conformal factor λ_x^c = 2 / (1 - c||x||^2).

        Uses `x.data` to avoid tracking gradients through this scaling
        coefficient (treated as a constant in the backward pass).
        """
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1.0 - self.c * x_sqnorm).clamp_min(self.min_norm)

    def _gyration(self, u: Tensor, v: Tensor, w: Tensor, dim: int = -1) -> Tensor:
        """Gyration operator gyr[u, v]w used in parallel transport."""
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = self.c**2
        a = -c2 * uw * v2 + self.c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - self.c * uw
        d = 1 + 2 * self.c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    # ------------------------------------------------------------------
    # Core manifold operations
    # ------------------------------------------------------------------

    def sqdist(self, p1: Tensor, p2: Tensor) -> Tensor:
        """Squared geodesic distance (Eq. 8).

        d_c(x,y)^2 = (2/sqrt(c) * artanh(sqrt(c) * ||-x ⊕_c y||))^2
        """
        sqrt_c = self.c**0.5
        dist_c = artanh(sqrt_c * self.mobius_add(-p1, p2, dim=-1).norm(dim=-1, p=2, keepdim=False))
        return (dist_c * 2 / sqrt_c) ** 2

    def egrad2rgrad(self, p: Tensor, dp: Tensor) -> Tensor:
        """Scale Euclidean gradient to Riemannian gradient: dp / λ_p^2."""
        dp /= self._lambda_x(p).pow(2)
        return dp

    def proj(self, x: Tensor) -> Tensor:
        """Project x onto the open ball by clipping points outside the boundary."""
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (self.c**0.5)
        return torch.where(norm > maxnorm, x / norm * maxnorm, x)

    def proj_tan(self, u: Tensor, p: Tensor) -> Tensor:
        """Tangent space projection (identity for Poincaré ball)."""
        return u

    def proj_tan0(self, u: Tensor) -> Tensor:
        """Tangent space projection at origin (identity for Poincaré ball)."""
        return u

    def expmap(self, u: Tensor, p: Tensor) -> Tensor:
        """Exponential map at p: follow geodesic from p in direction u (Eq. 12).

        exp_p^c(u) = p ⊕_c [ tanh(sqrt(c)/2 * λ_p^c * ||u||) * u / (sqrt(c) * ||u||) ]
        """
        sqrt_c = self.c**0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = tanh(sqrt_c / 2 * self._lambda_x(p) * u_norm) * u / (sqrt_c * u_norm)
        return self.mobius_add(p, second_term)

    def logmap(self, p1: Tensor, p2: Tensor) -> Tensor:
        """Logarithmic map at p1: tangent vector pointing from p1 to p2 (Eq. 12)."""
        sub = self.mobius_add(-p1, p2)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1)
        sqrt_c = self.c**0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u: Tensor) -> Tensor:
        """Exponential map at the origin (Eq. 12 with p=0, λ_0^c=2).

        exp_0^c(u) = tanh(sqrt(c) * ||u||) * u / (sqrt(c) * ||u||)
        """
        sqrt_c = self.c**0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        return tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

    def logmap0(self, p: Tensor) -> Tensor:
        """Logarithmic map at the origin (Eq. 12 with base=0, λ_0^c=2)."""
        sqrt_c = self.c**0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        return 1.0 / sqrt_c * artanh(sqrt_c * p_norm) / p_norm * p

    # ------------------------------------------------------------------
    # Möbius operations
    # ------------------------------------------------------------------

    def mobius_add(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        """Möbius addition x ⊕_c y (Eq. 6).

        (x ⊕_c y) = [(1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y]
                    / [1 + 2c<x,y> + c^2 ||x||^2 ||y||^2]
        """
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m: Tensor, x: Tensor) -> Tensor:
        """Möbius matrix-vector multiplication M ⊗_c x (Ganea 2018, Lemma 6).

        M ⊗_c x = (1/sqrt(c)) * tanh(||Mx||/||x|| * artanh(sqrt(c)*||x||)) * Mx/||Mx||
        """
        sqrt_c = self.c**0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True).bool()
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        return torch.where(cond, res_0, res_c)

    def init_weights(self, w: Tensor, irange: float = 1e-5) -> Tensor:
        w.data.uniform_(-irange, irange)
        return w

    # ------------------------------------------------------------------
    # Parallel transport
    # ------------------------------------------------------------------

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Parallel transport of u from x to y via gyration (Ungar 2008).

        P_{x→y}(u) = gyr[y, -x](u) * λ_x^c / λ_y^c
        """
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y

    def ptransp_(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Alias for ptransp (in-place-named variant, kept for compatibility)."""
        return self.ptransp(x, y, u)

    def ptransp0(self, x: Tensor, u: Tensor) -> Tensor:
        """Parallel transport from origin to x (Ganea 2018, Theorem 4, Eq. 16).

        P_{0→x}^c(u) = (λ_0^c / λ_x^c) * u = (2 / λ_x^c) * u
        """
        lambda_x = self._lambda_x(x)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def transp(self, x: Tensor, y: Tensor, v: Tensor, *, dim: int = -1) -> Tensor:
        return self.ptransp(x, y, v)

    def retr(self, x: Tensor, u: Tensor, *, dim: int = -1) -> Tensor:
        """First-order retraction (approximation of expmap for optimizer efficiency)."""
        return self.proj(x + u)

    def retr_transp(self, x: Tensor, u: Tensor, v: Tensor, *, dim: int = -1) -> tuple[Tensor, Tensor]:
        """Retract x by u and simultaneously transport v to the new point."""
        y = self.retr(x, u, dim=dim)
        return y, self.transp(x, y, v, dim=dim)

    # ------------------------------------------------------------------
    # Riemannian inner products
    # ------------------------------------------------------------------

    def inner(
        self,
        x: Tensor,
        u: Tensor,
        v: Tensor | None = None,
        *,
        keepdim: bool = False,
        dim: int = -1,
    ) -> Tensor:
        """Riemannian inner product at x: λ_x^2 * <u, v>."""
        if v is None:
            v = u
        return self._lambda_x(x) ** 2 * (u * v).sum(dim=dim, keepdim=keepdim)

    def component_inner(self, x: Tensor, u: Tensor, v: Tensor | None = None) -> Tensor:
        """Per-component Riemannian inner product (used by Riemannian Adam)."""
        return self.inner(x, u, v, keepdim=True)

    # ------------------------------------------------------------------
    # Coordinate change
    # ------------------------------------------------------------------

    def to_hyperboloid(self, x: Tensor) -> Tensor:
        """Map a point from the Poincaré ball to the hyperboloid model."""
        assert x.dim() == 2, "Input must be 2D (batch, dim)"
        K = 1.0 / self.c
        sqrtK = K**0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)
