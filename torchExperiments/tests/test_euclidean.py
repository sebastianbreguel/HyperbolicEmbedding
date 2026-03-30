"""Tests for the Euclidean manifold."""

from __future__ import annotations

import pytest
import torch

from manifolds.euclidean import Euclidean


class TestEuclideanSqdist:
    def test_self_distance_zero(self, euclidean: Euclidean):
        x = torch.randn(5, 3)
        dist = euclidean.sqdist(x, x)
        torch.testing.assert_close(dist, torch.zeros(5), atol=1e-6, rtol=1e-6)

    def test_symmetry(self, euclidean: Euclidean):
        x = torch.randn(4, 3)
        y = torch.randn(4, 3)
        torch.testing.assert_close(euclidean.sqdist(x, y), euclidean.sqdist(y, x))

    def test_non_negative(self, euclidean: Euclidean):
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)
        assert (euclidean.sqdist(x, y) >= 0).all()

    def test_known_value(self, euclidean: Euclidean):
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[3.0, 4.0]])
        assert euclidean.sqdist(x, y).item() == pytest.approx(25.0)


class TestEuclideanExpLog:
    def test_expmap_logmap_roundtrip(self, euclidean: Euclidean):
        p = torch.randn(4, 3)
        q = torch.randn(4, 3)
        u = euclidean.logmap(p, q)
        q_recovered = euclidean.expmap(u, p)
        torch.testing.assert_close(q_recovered, q, atol=1e-5, rtol=1e-5)

    def test_expmap0_is_identity(self, euclidean: Euclidean):
        u = torch.randn(3, 4)
        torch.testing.assert_close(euclidean.expmap0(u), u)

    def test_logmap0_is_identity(self, euclidean: Euclidean):
        p = torch.randn(3, 4)
        torch.testing.assert_close(euclidean.logmap0(p), p)


class TestEuclideanMobius:
    def test_mobius_add_is_addition(self, euclidean: Euclidean):
        x = torch.randn(4, 3)
        y = torch.randn(4, 3)
        torch.testing.assert_close(euclidean.mobius_add(x, y), x + y)

    def test_mobius_matvec_is_matmul(self, euclidean: Euclidean):
        m = torch.randn(5, 3)
        x = torch.randn(4, 3)
        expected = x @ m.T
        torch.testing.assert_close(euclidean.mobius_matvec(m, x), expected)


class TestEuclideanTransport:
    def test_ptransp_is_identity(self, euclidean: Euclidean):
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        v = torch.randn(3, 4)
        torch.testing.assert_close(euclidean.ptransp(x, y, v), v)

    def test_retr_transp(self, euclidean: Euclidean):
        x = torch.randn(3, 4)
        u = torch.randn(3, 4)
        v = torch.randn(3, 4)
        new_x, new_v = euclidean.retr_transp(x, u, v)
        torch.testing.assert_close(new_x, x + u)
        torch.testing.assert_close(new_v, v)


class TestEuclideanComponentInner:
    def test_self_inner(self, euclidean: Euclidean):
        x = torch.randn(3, 4)
        u = torch.randn(3, 4)
        result = euclidean.component_inner(x, u)
        torch.testing.assert_close(result, u.pow(2))

    def test_cross_inner(self, euclidean: Euclidean):
        x = torch.randn(3, 4)
        u = torch.randn(3, 4)
        v = torch.randn(3, 4)
        result = euclidean.component_inner(x, u, v)
        torch.testing.assert_close(result, u * v)
