import torch
import torch.nn as nn
import torch.nn.init as init
import math


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, 60, c, use_bias)
        self.linea2 = HypLinear(manifold, 60, 10, c, use_bias)
        self.linea3 = HypLinear(manifold, 10, out_features, c, use_bias)
        self.softmax = nn.Softmax(dim=1)
        # self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.linea2.forward(h)
        h = self.linea3.forward(h)
        ouput = self.softmax(h)
        return ouput

    def predict(self, test_x):
        h = self.linear.forward(test_x)
        h = self.linea2.forward(h)
        h = self.linea3.forward(h)
        ouput = self.softmax(h)
        return ouput

        # h = self.hyp_act.forward(h)


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
