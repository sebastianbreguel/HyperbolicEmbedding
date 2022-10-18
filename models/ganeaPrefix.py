import torch.nn as nn
from manifolds.base import ManifoldParameter
import torch.nn.functional as F

from layers import Linear, HypLinear, HypAct, HyperbolicMLR

from torch.nn import Linear


class Ganea(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, use_bias, hidden):
        super(Ganea, self).__init__()
        self.manifold = manifold
        self.c = c
        self.linear1 = HypLinear(manifold, in_features, hidden, c, 0, use_bias)
        self.linear2 = HypLinear(manifold, hidden, out_features, c, 0, use_bias)

        self.softmax = nn.Softmax(dim=1)
        if self.c == 1:
            self.softmax = HyperbolicMLR(manifold, out_features, out_features, c)

        self.hyp_Relu = HypAct(manifold, c, c, nn.ReLU())
        self.hyp_Leaky_Relu = HypAct(manifold, c, c, nn.LeakyReLU())

    def forward(self, x):
        x = self.manifold.proj(self.manifold.expmap0(x))
        x = self.hyp_Leaky_Relu(self.linear1(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        x = self.manifold.proj(self.manifold.expmap0(x))
        x = self.hyp_Leaky_Relu(self.linear1(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x
