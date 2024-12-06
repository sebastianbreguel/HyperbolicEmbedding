"""Euclidean manifold."""

import torch

from .base import Manifold


class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self, c=None):
        super(Euclidean, self).__init__()
        self.c = None  # c is the curvature which is None for Euclidean
        self.name = "Euclidean"

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.0)
        return p

    def sqdist(self, p1, p2):
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp):
        return dp

    def proj(self, p):
        return p

    def proj_tan(self, u, p):
        return u

    def proj_tan0(self, u):
        return u

    def expmap(self, u, p):
        return p + u

    def logmap(self, p1, p2):
        return p2 - p1

    def expmap0(self, u):
        return u

    def logmap0(self, p):
        return p

    def mobius_add(self, x, y, dim=-1):
        return x + y

    def mobius_matvec(self, m, x):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def ptransp(self, x, y, v):
        return v

    def ptransp0(self, x, v):
        return x + v

    def retr(self, x, u):
        return x + u

    def retr_transp(self, x, u, v):
        y = self.retr(x, u)
        v_tranps = self.ptransp(x, y, v)
        return y, v_tranps

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        # it is possible to factorize the manifold
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        target_shape = torch.broadcast_shapes(x.shape, inner.shape)
        return inner.expand(target_shape)
