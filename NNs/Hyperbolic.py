import torch
import torch.nn as nn
import torch.nn.init as init
import math
from utils.data_Params import *
from Manifolds.base import ManifoldParameter


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, use_bias):
        super(HNNLayer, self).__init__()
        self.manifold = manifold
        self.c = c
        self.linear1 = HypLinear(manifold, in_features, 50, c, use_bias)
        # self.linear2 = HypLinear(manifold, 40, 13, c, use_bias)
        self.linear3 = HypLinear(manifold, 50, out_features, c, use_bias)
        self.softmax = nn.Softmax(dim=1)
        self.hyp_Relu = HypAct(manifold, c, c, nn.LeakyReLU())
    
    def one_rnn_transform(self, W, h, U, x, b):
        hyp_x = x
        if self.inputs_geom == 'eucl':
            hyp_x = self.manifold.expmap0(x, self.c)

        hyp_b = b
        if self.bias_geom == 'eucl':
            hyp_b = self.manifold.expmap0(b, self.c)

        W_otimes_h = self.manifold.mobius_matvec(W, h, self.c)
        U_otimes_x = self.manifold.mobius_matvec(U, hyp_x, self.c)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, self.c)
        result = self.manifold.mobius_add(Wh_plus_Ux, hyp_b, self.c)
        return result

    def forward(self, x):
        x = self.manifold.expmap0(x)#, self.c)
        x = self.hyp_Relu(self.linear1(x))
        # x = self.hyp_Relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return x


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
        # self.bias = ManifoldParameter(
        #     torch.Tensor(out_features), requires_grad=True, c=c, manifold=manifold
        # )
        # self.weight = ManifoldParameter(
        #     torch.Tensor(out_features, in_features),
        #     requires_grad=True,
        #     c=c,
        #     manifold=manifold,
        # self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        # self.weight = nn.Parameter(
        #     torch.Tensor(out_features, in_features), requires_grad=True
        # )
        self.bias = ManifoldParameter(torch.Tensor(out_features), requires_grad=True, manifold=self.manifold, c=self.c)
        self.weight = ManifoldParameter(
            torch.Tensor(out_features, in_features), requires_grad=True, manifold=self.manifold, c=self.c
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x)
        res = self.manifold.proj(mv)
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
        xt = self.act(self.manifold.logmap0(x))
        xt = self.manifold.proj_tan0(xt)
        return self.manifold.proj(self.manifold.expmap0(xt))

    def extra_repr(self):
        return "c_in={}, c_out={}".format(self.c_in, self.c_out)
