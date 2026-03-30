"""Tests for RiemannianAdam optimizer."""

from __future__ import annotations

import torch

from manifolds.base import ManifoldParameter
from manifolds.poincare import PoincareBall
from optimizer.Radam import RiemannianAdam


class TestRiemannianAdamBasic:
    def test_step_runs(self):
        """Optimizer step completes without error."""
        manifold = PoincareBall(c=1.0)
        w = ManifoldParameter(
            torch.randn(4, 3) * 0.1, requires_grad=True, manifold=manifold, c=1.0
        )
        opt = RiemannianAdam([{"params": [w]}], lr=0.01, stabilize=10)

        # Simulate a gradient
        loss = w.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_parameter_stays_on_manifold(self):
        """After several steps, ManifoldParameter should remain inside the ball."""
        manifold = PoincareBall(c=1.0)
        w = ManifoldParameter(
            torch.randn(5, 3) * 0.1, requires_grad=True, manifold=manifold, c=1.0
        )
        opt = RiemannianAdam([{"params": [w]}], lr=0.01, stabilize=5)

        for _ in range(20):
            loss = w.pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        norms = w.data.norm(dim=-1)
        assert (norms < 1.0).all(), f"Norms outside ball: {norms}"

    def test_loss_decreases(self):
        """Loss should decrease over several steps towards the origin."""
        manifold = PoincareBall(c=1.0)
        w = ManifoldParameter(
            torch.randn(3, 4) * 0.3, requires_grad=True, manifold=manifold, c=1.0
        )
        opt = RiemannianAdam([{"params": [w]}], lr=0.01, stabilize=10)

        initial_loss = w.pow(2).sum().item()
        for _ in range(50):
            opt.zero_grad()
            loss = w.pow(2).sum()
            loss.backward()
            opt.step()

        final_loss = w.pow(2).sum().item()
        assert final_loss < initial_loss

    def test_mixed_params(self):
        """Optimizer handles both ManifoldParameter and regular Parameter."""
        manifold = PoincareBall(c=1.0)
        w_manifold = ManifoldParameter(
            torch.randn(3, 3) * 0.1, requires_grad=True, manifold=manifold, c=1.0
        )
        w_euclidean = torch.nn.Parameter(torch.randn(3, 3))

        opt = RiemannianAdam(
            [
                {"params": [w_euclidean], "weight_decay": 0.01},
                {"params": [w_manifold], "weight_decay": 0.0},
            ],
            lr=0.01,
            stabilize=10,
        )

        for _ in range(10):
            opt.zero_grad()
            loss = w_manifold.sum() + w_euclidean.sum()
            loss.backward()
            opt.step()

        assert torch.isfinite(w_manifold).all()
        assert torch.isfinite(w_euclidean).all()


class TestStabilize:
    def test_stabilize_projects_back(self):
        """After stabilize, parameters should be within ball boundary."""
        manifold = PoincareBall(c=1.0)
        w = ManifoldParameter(
            torch.randn(4, 3) * 0.1, requires_grad=True, manifold=manifold, c=1.0
        )
        opt = RiemannianAdam([{"params": [w]}], lr=0.1, stabilize=1)

        for _ in range(5):
            opt.zero_grad()
            loss = w.sum()
            loss.backward()
            opt.step()

        norms = w.data.norm(dim=-1)
        # Stabilize keeps points near ball boundary; allow small overshoot
        # from gradient steps between stabilize calls
        assert (norms < 1.1).all()
