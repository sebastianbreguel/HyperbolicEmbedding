"""Tests for hyperbolic layers: HypLinear, HypAct, HyperbolicMLR."""

from __future__ import annotations

import torch

from layers.hyp_layers import HypAct, HypLinear
from layers.hyp_softmax import HyperbolicMLR, mobius_addition_batch, tensor_dot
from manifolds.euclidean import Euclidean
from manifolds.poincare import PoincareBall


class TestHypLinear:
    def test_output_shape(self, poincare: PoincareBall):
        layer = HypLinear(poincare, in_features=10, out_features=5, c=1.0, dropout=0.0)
        x = poincare.proj(torch.randn(4, 10) * 0.1)
        out = layer(x)
        assert out.shape == (4, 5)

    def test_output_finite(self, poincare: PoincareBall):
        layer = HypLinear(poincare, in_features=8, out_features=4, c=1.0, dropout=0.0)
        x = poincare.proj(torch.randn(6, 8) * 0.1)
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_output_in_ball(self, poincare: PoincareBall):
        layer = HypLinear(poincare, in_features=8, out_features=4, c=1.0, dropout=0.0)
        x = poincare.proj(torch.randn(6, 8) * 0.1)
        out = layer(x)
        norms = out.norm(dim=-1)
        assert (norms < 1.0 / poincare.c ** 0.5 + 0.01).all()

    def test_gradient_flows(self, poincare: PoincareBall):
        layer = HypLinear(poincare, in_features=5, out_features=3, c=1.0, dropout=0.0)
        x = poincare.proj(torch.randn(2, 5) * 0.1)
        x.requires_grad_(True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_euclidean_mode(self, euclidean: Euclidean):
        layer = HypLinear(euclidean, in_features=5, out_features=3, c=0.0, dropout=0.0)
        x = torch.randn(4, 5)
        out = layer(x)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()


class TestHypAct:
    def test_output_shape(self, poincare: PoincareBall):
        act = HypAct(poincare, c_in=1.0, c_out=1.0, act=torch.nn.ReLU())
        x = poincare.proj(poincare.expmap0(torch.randn(4, 5) * 0.3))
        out = act(x)
        assert out.shape == (4, 5)

    def test_output_in_ball(self, poincare: PoincareBall):
        act = HypAct(poincare, c_in=1.0, c_out=1.0, act=torch.nn.ReLU())
        x = poincare.proj(poincare.expmap0(torch.randn(8, 5) * 0.3))
        out = act(x)
        norms = out.norm(dim=-1)
        assert (norms < 1.0 / poincare.c ** 0.5 + 0.01).all()

    def test_output_finite(self, poincare: PoincareBall):
        act = HypAct(poincare, c_in=1.0, c_out=1.0, act=torch.nn.ReLU())
        x = poincare.proj(poincare.expmap0(torch.randn(4, 3) * 0.3))
        out = act(x)
        assert torch.isfinite(out).all()


class TestHyperbolicMLR:
    def test_output_shape(self, poincare: PoincareBall):
        mlr = HyperbolicMLR(poincare, ball_dim=4, n_classes=3, c=1.0)
        x = poincare.proj(poincare.expmap0(torch.randn(8, 4) * 0.3))
        out = mlr(x)
        assert out.shape == (8, 3)

    def test_output_finite(self, poincare: PoincareBall):
        mlr = HyperbolicMLR(poincare, ball_dim=5, n_classes=2, c=1.0)
        x = poincare.proj(poincare.expmap0(torch.randn(6, 5) * 0.2))
        out = mlr(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, poincare: PoincareBall):
        mlr = HyperbolicMLR(poincare, ball_dim=4, n_classes=3, c=1.0)
        x = poincare.proj(poincare.expmap0(torch.randn(4, 4) * 0.2))
        x.requires_grad_(True)
        out = mlr(x)
        out.sum().backward()
        assert x.grad is not None


class TestTensorDot:
    def test_shape(self):
        x = torch.randn(5, 3)
        y = torch.randn(4, 3)
        result = tensor_dot(x, y)
        assert result.shape == (5, 4)

    def test_known_value(self):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = tensor_dot(x, y)
        expected = torch.tensor([[1.0, 0.0]])
        torch.testing.assert_close(result, expected)


class TestMobiusAdditionBatch:
    def test_output_shape(self):
        x = torch.randn(8, 4) * 0.3
        y = torch.randn(3, 4) * 0.3
        c = torch.tensor(1.0)
        result = mobius_addition_batch(x, y, c)
        assert result.shape == (8, 3, 4)

    def test_finite(self):
        x = torch.randn(5, 3) * 0.2
        y = torch.randn(4, 3) * 0.2
        c = torch.tensor(1.0)
        result = mobius_addition_batch(x, y, c)
        assert torch.isfinite(result).all()
