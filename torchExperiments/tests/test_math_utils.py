"""Tests for manifolds/math_utils.py — numerically stable inverse hyperbolics."""

from __future__ import annotations

import torch

from manifolds.math_utils import arcosh, arsinh, artanh, cosh, sinh, tanh


class TestArtanh:
    def test_roundtrip_with_tanh(self):
        """artanh(tanh(x)) ≈ x for moderate x."""
        x = torch.linspace(-2.0, 2.0, 50)
        result = artanh(tanh(x))
        torch.testing.assert_close(result, x, atol=1e-4, rtol=1e-4)

    def test_near_boundary(self):
        """artanh should not produce inf/nan near ±1."""
        x = torch.tensor([0.999, -0.999, 0.99999])
        result = artanh(x)
        assert torch.isfinite(result).all()

    def test_zero(self):
        """artanh(0) = 0."""
        result = artanh(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([0.0]), atol=1e-6, rtol=1e-6)

    def test_gradient_exists(self):
        """artanh should be differentiable."""
        x = torch.tensor([0.3, -0.5], requires_grad=True)
        y = artanh(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_value(self):
        """d/dx artanh(x) = 1/(1-x^2)."""
        x = torch.tensor([0.3, -0.5], requires_grad=True)
        y = artanh(x).sum()
        y.backward()
        expected = 1.0 / (1 - x.data ** 2)
        torch.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


class TestArsinh:
    def test_roundtrip_with_sinh(self):
        """arsinh(sinh(x)) ≈ x."""
        x = torch.linspace(-5.0, 5.0, 50)
        result = arsinh(sinh(x))
        torch.testing.assert_close(result, x, atol=1e-3, rtol=1e-3)

    def test_zero(self):
        result = arsinh(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([0.0]), atol=1e-6, rtol=1e-6)

    def test_gradient_exists(self):
        x = torch.tensor([1.0, -2.0], requires_grad=True)
        y = arsinh(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_value(self):
        """d/dx arsinh(x) = 1/sqrt(1+x^2)."""
        x = torch.tensor([1.0, -2.0], requires_grad=True)
        y = arsinh(x).sum()
        y.backward()
        expected = 1.0 / (1 + x.data ** 2) ** 0.5
        torch.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


class TestArcosh:
    def test_roundtrip_with_cosh(self):
        """arcosh(cosh(x)) ≈ |x| for x > 0."""
        x = torch.linspace(0.5, 5.0, 30)
        result = arcosh(cosh(x))
        torch.testing.assert_close(result, x, atol=1e-3, rtol=1e-3)

    def test_near_one(self):
        """arcosh(1) ≈ 0 (clamped slightly above 1)."""
        result = arcosh(torch.tensor([1.0]))
        assert torch.isfinite(result).all()
        assert result.item() < 0.01

    def test_gradient_exists(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = arcosh(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestClampedFunctions:
    def test_cosh_large_input(self):
        x = torch.tensor([100.0, -100.0])
        result = cosh(x)
        assert torch.isfinite(result).all()

    def test_sinh_large_input(self):
        x = torch.tensor([100.0, -100.0])
        result = sinh(x)
        assert torch.isfinite(result).all()

    def test_tanh_large_input(self):
        x = torch.tensor([100.0, -100.0])
        result = tanh(x)
        assert torch.isfinite(result).all()
        # tanh saturates to ±1
        torch.testing.assert_close(result.abs(), torch.ones(2), atol=1e-4, rtol=1e-4)
