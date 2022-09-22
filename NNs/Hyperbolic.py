import torch
import torch.nn as nn
import torch.nn.init as init
import math
from utils.data_Params import *


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, use_bias):
        super(HNNLayer, self).__init__()
        self.linear1 = HypLinear(manifold, in_features, 8 * LARGE, c, use_bias)
        self.linear2 = HypLinear(manifold, 8 * LARGE, 4 * LARGE, c, use_bias)
        self.linear3 = HypLinear(manifold, 4 * LARGE, out_features, c, use_bias)
        self.softmax = nn.Softmax(dim=1)
        self.hyp_Relu = HypAct(manifold, c, c, nn.LeakyReLU())


    def forward(self, x):
        x = self.hyp_Relu(self.linear1(x))
        x = self.hyp_Relu(self.linear2(x))
        x = self.softmax(self.linear3(x))
        return x

    def predict(self, test_x):
        test_x = self.forward(test_x)
        return test_x


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
        hyp_bias = self.manifold.expmap0(bias, self.c)
        hyp_bias = self.manifold.proj(hyp_bias, self.c)
        res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
        res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return "in_features={}, out_features={}, c={}".format(
            self.in_features, self.out_features, self.c
        )


class HypAct(nn.Module):
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
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return "c_in={}, c_out={}".format(self.c_in, self.c_out)
