"""Poincare ball manifold."""

import itertools

import torch

from .base import Manifold
from .math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.
    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c
    Note that 1/sqrt(c) is the Poincare ball radius.
    """

    def __init__(self, c):
        super(PoincareBall, self).__init__()
        self.name = "PoincareBall"
        self.min_norm = 1e-15
        self.c = c
        self.eps = {torch.float32: 4e-3, torch.float32: 1e-5}

    def sqdist(self, p1, p2):
        sqrt_c = self.c**0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist**2

    def _lambda_x(self, x):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1.0 - self.c * x_sqnorm).clamp_min(self.min_norm)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1):
        return self.ptransp(x, y, v)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        # always assume u is scaled properly
        approx = x + u
        return self.proj(approx)

    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1):
        y = self.retr(x, u, dim=dim)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def egrad2rgrad(self, p, dp):
        lambda_p = self._lambda_x(p)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (self.c**0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p):
        return u

    def proj_tan0(self, u):
        return u

    def expmap(self, u, p):
        sqrt_c = self.c**0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
            tanh(sqrt_c / 2 * self._lambda_x(p) * u_norm) * u / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term)
        return gamma_1

    def logmap(self, p1, p2):
        sub = self.mobius_add(-p1, p2)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1)
        sqrt_c = self.c**0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u):
        sqrt_c = self.c**0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p):
        sqrt_c = self.c**0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1.0 / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x):
        sqrt_c = self.c**0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = (
            tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        )
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.type(torch.BoolTensor), res_0, res_c)
        return res

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, dim: int = -1):
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

    def ptransp(self, x, y, u):
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y

    def ptransp_(self, x, y, u):
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y

    def ptransp0(self, x, u):
        lambda_x = self._lambda_x(x)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x):
        K = 1.0 / self.c
        sqrtK = K**0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        if v is None:
            v = u

        return self._lambda_x(x) ** 2 * (u * v).sum(dim=dim, keepdim=keepdim)

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        return self.inner(x, u, v, keepdim=True)
