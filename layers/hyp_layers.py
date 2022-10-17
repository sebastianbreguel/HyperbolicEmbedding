"""Hyperbolic layers."""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from manifolds.base import ManifoldParameter
from manifolds.math_utils import arsinh


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ["lp", "rec"]:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.0])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(
            manifold, in_features, out_features, c, dropout, use_bias, scale=10
        )
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = ManifoldParameter(
            torch.Tensor(out_features), requires_grad=True, c=c, manifold=manifold
        )
        self.weight = ManifoldParameter(
            torch.Tensor(out_features, in_features),
            requires_grad=True,
            c=c,
            manifold=manifold,
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x)
        res = self.manifold.proj(mv)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1))
            hyp_bias = self.manifold.expmap0(bias)
            hyp_bias = self.manifold.proj(hyp_bias)
            res = self.manifold.mobius_add(res, hyp_bias)
            res = self.manifold.proj(res)
        return res

    def extra_repr(self):
        return "in_features={}, out_features={}, c={}".format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x))
        xt = self.manifold.proj_tan0(xt)
        return self.manifold.proj(self.manifold.expmap0(xt))

    def extra_repr(self):
        return "c_in={}, c_out={}".format(self.c_in, self.c_out)
