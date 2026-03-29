"""Numerically stable implementations of inverse hyperbolic functions.

All custom autograd functions compute forward passes in float64 for precision,
then cast back to the input dtype. Clamping prevents domain errors at boundaries.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cosh(x: Tensor, clamp: float = 15) -> Tensor:
    """Clamped cosh to avoid overflow."""
    return x.clamp(-clamp, clamp).cosh()


def sinh(x: Tensor, clamp: float = 15) -> Tensor:
    """Clamped sinh to avoid overflow."""
    return x.clamp(-clamp, clamp).sinh()


def tanh(x: Tensor, clamp: float = 15) -> Tensor:
    """Clamped tanh to avoid saturation in extreme values."""
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x: Tensor) -> Tensor:
    """Inverse hyperbolic cosine: defined for x >= 1."""
    return Arcosh.apply(x)


def arsinh(x: Tensor) -> Tensor:
    """Inverse hyperbolic sine: defined for all real x."""
    return Arsinh.apply(x)


def artanh(x: Tensor) -> Tensor:
    """Inverse hyperbolic tangent: defined for x in (-1, 1)."""
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    """Numerically stable artanh with correct gradient.

    Input is clamped to (-1+eps, 1-eps) to stay in the valid domain.
    Forward pass uses float64 to avoid precision loss near the boundary.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input**2)


class Arsinh(torch.autograd.Function):
    """Numerically stable arsinh with correct gradient.

    Uses the identity arsinh(x) = log(x + sqrt(1 + x^2)).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input**2) ** 0.5


class Arcosh(torch.autograd.Function):
    """Numerically stable arcosh with correct gradient.

    Uses the identity arcosh(x) = log(x + sqrt(x^2 - 1)). Input clamped to [1+eps, inf).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x.clamp(min=1.0 + 1e-5)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-5).log_().to(x.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        (input,) = ctx.saved_tensors
        return grad_output / (input**2 - 1) ** 0.5
