"""Riemannian Adam optimiser for parameters on Riemannian manifolds.

Implements the Riemannian Adam algorithm from Bécigneul & Ganea 2019
"Riemannian Adaptive Optimization Methods" (https://arxiv.org/abs/1810.00760).

The key difference from standard Adam is that gradient accumulation and
parameter updates are performed using manifold-aware operations:
  - egrad2rgrad: converts Euclidean gradient to Riemannian gradient
  - retr_transp: retraction + parallel transport of momentum
"""

from __future__ import annotations

from typing import Any

import torch
from geoopt import Euclidean, ManifoldParameter


def proju(x: torch.Tensor, u: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project u onto the tangent space at x (broadcast-safe identity for flat space)."""
    target_shape = torch.broadcast_shapes(x.shape, u.shape)
    return u.expand(target_shape)


class OptimMixin:
    """Mixin that adds Riemannian manifold awareness to a standard PyTorch optimizer."""

    _default_manifold = Euclidean()

    def __init__(self, *args: Any, stabilize: int | None = None, **kwargs: Any) -> None:
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        param_group.setdefault("stabilize", self._stabilize)
        return super().add_param_group(param_group)  # type: ignore[misc]

    def stabilize_group(self, group: dict[str, Any]) -> None:
        """Stabilize one parameter group (no-op in base class)."""

    def stabilize(self) -> None:
        """Project all ManifoldParameters back onto their manifold.

        Useful when numerical drift has pushed parameters slightly off-manifold
        (e.g. outside the Poincaré ball).
        """
        for group in self.param_groups:
            self.stabilize_group(group)


def copy_or_set_(dest: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Copy ``source`` into ``dest``, respecting strides.

    A workaround for https://github.com/geoopt/geoopt/issues/70: when strides
    differ, ``set_`` would corrupt the tensor, so we fall back to ``copy_``.

    Args:
        dest: Destination tensor (modified in-place).
        source: Source data.

    Returns:
        ``dest`` after the copy.
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


__all__ = ["RiemannianAdam"]


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)


    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self, *args: Any, stabilize: int, **kwargs: Any) -> None:
        super().__init__(*args, stabilize=stabilize, **kwargs)

    def step(self, closure: Any | None = None) -> torch.Tensor | None:
        """Perform a single optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns the loss.

        Returns:
            The loss evaluated by the closure, or None.
        """
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                group["step"] += 1
                # group['max_grad_norm'] = self.max_grad_norm
                for point in group["params"]:
                    # if group['max_grad_norm'] > 0:
                    #     clip_grad_norm_(point, group['max_grad_norm'])
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter)):
                        manifold = point.manifold

                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    if isinstance(point, (ManifoldParameter)):
                        grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq
                    else:
                        denom = exp_avg_sq
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    step_size = learning_rate

                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = (exp_avg / bias_correction1) / (
                        (denom / bias_correction2).sqrt() + eps
                    )
                    if not isinstance(point, (ManifoldParameter)):
                        direction = direction + point * weight_decay
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -step_size * direction, exp_avg
                    )
                    # use copy only for user facing point
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group: dict[str, Any]) -> None:
        """Project ManifoldParameters in one group back onto their manifold."""
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.proj(p))
            exp_avg.copy_(proju(p, exp_avg))
