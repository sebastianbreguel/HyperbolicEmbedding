"""Abstract base class for Riemannian manifolds."""

from __future__ import annotations

from geoopt import ManifoldTensor
from torch import Tensor
from torch.nn import Module, Parameter


class Manifold(Module):
    """Abstract base class defining the interface for a Riemannian manifold.

    All manifold implementations (Euclidean, PoincareBall, …) must subclass
    this and implement every method.
    """

    def __init__(self) -> None:
        super().__init__()
        self.eps: float = 10e-4

    def sqdist(self, p1: Tensor, p2: Tensor) -> Tensor:
        """Squared geodesic distance between corresponding points p1 and p2."""
        raise NotImplementedError

    def egrad2rgrad(self, p: Tensor, dp: Tensor) -> Tensor:
        """Convert Euclidean gradient dp at point p to Riemannian gradient."""
        raise NotImplementedError

    def proj(self, p: Tensor) -> Tensor:
        """Project p onto the manifold (clip to valid region)."""
        raise NotImplementedError

    def proj_tan(self, u: Tensor, p: Tensor) -> Tensor:
        """Project u onto the tangent space at p."""
        raise NotImplementedError

    def proj_tan0(self, u: Tensor) -> Tensor:
        """Project u onto the tangent space at the origin."""
        raise NotImplementedError

    def expmap(self, u: Tensor, p: Tensor) -> Tensor:
        """Exponential map: move from p in the direction of tangent vector u."""
        raise NotImplementedError

    def logmap(self, p1: Tensor, p2: Tensor) -> Tensor:
        """Logarithmic map: tangent vector at p1 pointing toward p2."""
        raise NotImplementedError

    def expmap0(self, u: Tensor) -> Tensor:
        """Exponential map at the origin."""
        raise NotImplementedError

    def logmap0(self, p: Tensor) -> Tensor:
        """Logarithmic map at the origin: tangent vector at 0 pointing toward p."""
        raise NotImplementedError

    def mobius_add(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        """Möbius addition of x and y (generalizes vector addition to the manifold)."""
        raise NotImplementedError

    def mobius_matvec(self, m: Tensor, x: Tensor) -> Tensor:
        """Möbius matrix-vector multiplication (Ganea et al. 2018, Lemma 6)."""
        raise NotImplementedError

    def init_weights(self, w: Tensor, irange: float = 1e-5) -> Tensor:
        """Initialize weights uniformly in [-irange, irange]."""
        raise NotImplementedError

    def component_inner(self, p: Tensor, u: Tensor, v: Tensor | None = None) -> Tensor:
        """Per-component Riemannian inner product at p (used by Riemannian Adam)."""
        raise NotImplementedError

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Parallel transport of tangent vector u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x: Tensor, u: Tensor) -> Tensor:
        """Parallel transport of tangent vector u from the origin to x."""
        raise NotImplementedError


class ManifoldParameter(ManifoldTensor, Parameter):
    """A :class:`torch.nn.Parameter` that lives on a Riemannian manifold.

    Used by :class:`~layers.HypLinear` so that the Riemannian Adam optimizer
    can identify which parameters require Riemannian gradient updates.
    """

    def __new__(cls, data: Tensor, requires_grad: bool, manifold: Manifold, c: float) -> ManifoldParameter:
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data: Tensor, requires_grad: bool, manifold: Manifold, c: float) -> None:
        self.c = c
        self.manifold = manifold

    def __repr__(self) -> str:
        return f"{self.manifold.name} Parameter containing:\n" + super(Parameter, self).__repr__()
