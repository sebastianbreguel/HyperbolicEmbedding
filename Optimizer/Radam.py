"""Riemannian adam optimizer geoopt implementation (https://github.com/geoopt/)."""
import torch.optim
from Manifolds.euclidean import Euclidean
from Manifolds.base import ManifoldParameter
from geoopt import ManifoldTensor

# in order not to create it at each iteration


class OptimMixin(object):
    _default_manifold = Euclidean()

    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def add_param_group(self, param_group: dict):
        param_group.setdefault("stabilize", self._stabilize)
        return super().add_param_group(param_group)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons."""
        for group in self.param_groups:
            self.stabilize_group(group)


def copy_or_set_(dest, source):
    """
    A workaround to respect strides of :code:`dest` when copying :code:`source`
    (https://github.com/geoopt/geoopt/issues/70)
    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor
    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


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

    def step(self, closure=None):
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
                for point in group["params"]:
                    grad = point.grad
                    # print(grad)
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                        c = point.c
                        # print("yapo rey")
                    else:
                        # print("tsiuu")
                        # manifold = PoincareBall()
                        # c = 1
                        c = None
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
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
                    print(exp_avg_sq)
                    # actual step
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    print(betas[1])
                    print(
                        exp_avg_sq.mul_(betas[1]).shape,
                        "hola",
                        manifold.inner(point, grad).shape,
                    )
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.inner(point, grad), alpha=1 - betas[1]
                    )
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -learning_rate * direction, exp_avg
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

    # @torch.no_grad()
    # def stabilize_group(self, group):
    #     for p in group["params"]:
    #         if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
    #             continue
    #         state = self.state[p]
    #         if not state:  # due to None grads
    #             continue
    #         manifold = p.manifold
    #         exp_avg = state["exp_avg"]
    #         p.copy_(manifold.projx(p))
    #         exp_avg.copy_(manifold.proju(p, exp_avg))

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            c = p.c
            exp_avg = state["exp_avg"]
            copy_or_set_(p, manifold.proj(p))
            exp_avg.set_(manifold.proj_tan(p, exp_avg))
