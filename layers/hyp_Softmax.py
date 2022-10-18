"""
Implementation of various mathematical operations in the Poincare ball model of hyperbolic space. Some
functions are based on the implementation in https://github.com/geoopt/geoopt (copyright by Maxim Kochurov).
"""

import numpy as np
import torch
from scipy.special import gamma
import math
import torch.nn as nn
import torch.nn.init as init
from manifolds import artanh, arsinh



class HyperbolicMLR(nn.Module):
    """Performs softmax classification in Hyperbolic space."""

    def __init__(self, manifold, ball_dim, n_classes, c):
        """Initializes a HyperbolicMLR object.
        Args:
          ball_dim: Dimensionality of the embedding space.
          n_classes: Number of classes for training the network.
          c: Curvature of the Poincare ball.
        """
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.manifold = manifold
        self.reset_parameters()

    def forward(self, x):
        c = torch.as_tensor(self.c).type_as(x)
        p_vals_poincare = self.manifold.expmap0(self.p_vals)
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.a_vals, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.p_vals, a=np.sqrt(5))

def tensor_dot(x, y):
    """Performs a tensor dot product."""
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def mobius_addition_batch(x, y, c):
    """Performs a vectorized Mobius addition operator between x and y."""
    xy = tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c**2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res



def hyperbolic_softmax(x, a, p, c):
    """Computes the hyperbolic softmax function."""
    lambda_pkc = 2 / (1 - c * p.pow(2).sum(dim=1).clamp_max((1.0 / c) - 1e-4))
    k = lambda_pkc * torch.norm(a, dim=1).clamp_min(1e-7) / torch.sqrt(c)
    mob_add = mobius_addition_batch(-p, x, c)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * a.unsqueeze(1), dim=-1)
    denom = torch.norm(a, dim=1, keepdim=True).clamp_min(1e-7) * (
        1 - c * mob_add.pow(2).sum(dim=2)
    )
    logit = k.unsqueeze(1) * arsinh(num / denom)
    return logit.permute(1, 0)
