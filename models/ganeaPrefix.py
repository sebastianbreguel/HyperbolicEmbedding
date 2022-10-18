import torch.nn as nn
from manifolds.base import ManifoldParameter
import torch.nn.functional as F

from layers import Linear, HypLinear, HypAct, HyperbolicMLR



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
            self.softmax = HyperbolicMLR(manifold, 2, 2, c)

        self.hyp_Relu = HypAct(manifold, c, c, nn.LeakyReLU())

    def forward(self, x):
        # x = self.manifold.proj(self.manifold.expmap0(x))
        x = self.hyp_Relu(self.linear1(x))
        x = self.linear2(x)
        # x = self.manifold.logmap0(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        x = self.manifold.expmap0(x)
        x = self.hyp_Relu(self.linear1(x))
        x = self.linear2(x)
        x = self.manifold.logmap0(x)
        return x




# class Encoder(nn.Module):
#     """
#     Encoder abstract class.
#     """

#     def __init__(self, c):
#         super(Encoder, self).__init__()
#         self.c = c

#     def encode(self, x, adj):
#         if self.encode_graph:
#             input = (x, adj)
#             output, _ = self.layers.forward(input)
#         else:
#             output = self.layers.forward(x)
#         return output


# class MLP(Encoder):
#     """
#     Multi-layer perceptron.
#     """

#     def __init__(self, c, args):
#         super(MLP, self).__init__(c)
#         assert args.num_layers > 0
#         dims, acts = get_dim_act(args)
#         layers = []
#         for i in range(len(dims) - 1):
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
#         self.layers = nn.Sequential(*layers)
#         self.encode_graph = False


# class HNN(Encoder):
#     """
#     Hyperbolic Neural Networks.
#     """

#     def __init__(self, c, manifold, args):
#         super(HNN, self).__init__(c)
#         self.manifold = manifold
#         assert args.num_layers > 1
#         dims, acts, _ = hyp_layers.get_dim_act_curv(args)
#         hnn_layers = []
#         for i in range(len(dims) - 1):
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             hnn_layers.append(
#                 hyp_layers.HNNLayer(
#                     self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias
#                 )
#             )
#         self.layers = nn.Sequential(*hnn_layers)
#         self.encode_graph = False

    # def encode(self, x, adj):
    #     x_hyp = self.manifold.proj(
    #         self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c),
    #         c=self.c,
    #     )
    #     return super(HNN, self).encode(x_hyp, adj)

