"""Tests for the HNN model."""

from __future__ import annotations

import pytest
import torch

from manifolds.euclidean import Euclidean
from manifolds.poincare import PoincareBall
from models.hnn import HNN


class TestHNNHyperbolic:
    @pytest.fixture
    def hyp_model(self) -> HNN:
        manifold = PoincareBall(c=1.0)
        return HNN(manifold, in_features=10, out_features=2, c=1.0, hidden=16)

    def test_output_shape(self, hyp_model: HNN):
        x = torch.randn(4, 10) * 0.1
        out = hyp_model(x)
        assert out.shape == (4, 2)

    def test_output_finite(self, hyp_model: HNN):
        x = torch.randn(8, 10) * 0.1
        out = hyp_model(x)
        assert torch.isfinite(out).all()

    def test_gradient_flows_through(self, hyp_model: HNN):
        x = torch.randn(4, 10) * 0.1
        x.requires_grad_(True)
        x.retain_grad()
        out = hyp_model(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_train_eval_modes(self, hyp_model: HNN):
        x = torch.randn(2, 10) * 0.1
        hyp_model.train()
        out_train = hyp_model(x)
        hyp_model.eval()
        out_eval = hyp_model(x)
        assert out_train.shape == out_eval.shape


class TestHNNEuclidean:
    @pytest.fixture
    def euc_model(self) -> HNN:
        manifold = Euclidean()
        return HNN(manifold, in_features=10, out_features=3, c=0.0, hidden=16)

    def test_output_shape(self, euc_model: HNN):
        x = torch.randn(4, 10)
        out = euc_model(x)
        assert out.shape == (4, 3)

    def test_output_finite(self, euc_model: HNN):
        x = torch.randn(8, 10)
        out = euc_model(x)
        assert torch.isfinite(out).all()

    def test_raw_logit_output(self, euc_model: HNN):
        """Euclidean model outputs raw logits for CrossEntropyLoss compatibility."""
        x = torch.randn(4, 10)
        out = euc_model(x)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()


class TestHNNVariousSizes:
    @pytest.mark.parametrize("in_f,out_f,hidden", [
        (5, 2, 8),
        (20, 10, 32),
        (140, 6, 64),
    ])
    def test_hyperbolic_various_sizes(self, in_f: int, out_f: int, hidden: int):
        manifold = PoincareBall(c=1.0)
        model = HNN(manifold, in_f, out_f, c=1.0, hidden=hidden)
        x = torch.randn(4, in_f) * 0.1
        out = model(x)
        assert out.shape == (4, out_f)
        assert torch.isfinite(out).all()
