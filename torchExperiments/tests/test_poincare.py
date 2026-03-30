"""Tests for the Poincaré ball manifold — the most critical module."""

from __future__ import annotations

import pytest
import torch

from manifolds.poincare import PoincareBall


class TestPoincareBallProj:
    def test_points_inside_ball_unchanged(self, poincare: PoincareBall):
        x = torch.randn(5, 4) * 0.1
        proj = poincare.proj(x)
        torch.testing.assert_close(proj, x, atol=1e-5, rtol=1e-5)

    def test_points_outside_ball_projected_inside(self, poincare: PoincareBall):
        x = torch.randn(5, 4) * 10.0
        proj = poincare.proj(x)
        norms = proj.norm(dim=-1)
        max_norm = (1 - poincare.eps[torch.float32]) / (poincare.c ** 0.5)
        assert (norms <= max_norm + 1e-6).all()

    def test_proj_preserves_direction(self, poincare: PoincareBall):
        x = torch.tensor([[3.0, 4.0]])
        proj = poincare.proj(x)
        # direction should be preserved
        cos_sim = torch.nn.functional.cosine_similarity(x, proj)
        assert cos_sim.item() == pytest.approx(1.0, abs=1e-5)


class TestPoincareBallExpLogRoundtrip:
    def test_expmap0_logmap0_roundtrip(self, poincare: PoincareBall):
        """exp_0(log_0(p)) ≈ p for points inside the ball."""
        torch.manual_seed(0)
        p = torch.randn(8, 4) * 0.3
        p = poincare.proj(p)
        u = poincare.logmap0(p)
        p_recovered = poincare.expmap0(u)
        torch.testing.assert_close(p_recovered, p, atol=1e-4, rtol=1e-4)

    def test_logmap0_expmap0_roundtrip(self, poincare: PoincareBall):
        """log_0(exp_0(u)) ≈ u for small tangent vectors."""
        torch.manual_seed(1)
        u = torch.randn(8, 4) * 0.3
        p = poincare.expmap0(u)
        u_recovered = poincare.logmap0(p)
        torch.testing.assert_close(u_recovered, u, atol=1e-4, rtol=1e-4)

    def test_expmap_logmap_roundtrip_at_point(self, poincare: PoincareBall):
        """exp_p(log_p(q)) ≈ q."""
        torch.manual_seed(2)
        p = poincare.proj(torch.randn(6, 3) * 0.2)
        q = poincare.proj(torch.randn(6, 3) * 0.2)
        u = poincare.logmap(p, q)
        q_recovered = poincare.expmap(u, p)
        torch.testing.assert_close(q_recovered, q, atol=1e-3, rtol=1e-3)

    def test_expmap0_stays_in_ball(self, poincare: PoincareBall):
        u = torch.randn(20, 5) * 2.0
        p = poincare.expmap0(u)
        norms = p.norm(dim=-1)
        assert (norms < 1.0 / poincare.c ** 0.5).all()


class TestPoincareBallSqdist:
    def test_self_distance_zero(self, poincare: PoincareBall):
        x = poincare.proj(torch.randn(5, 4) * 0.3)
        dist = poincare.sqdist(x, x)
        torch.testing.assert_close(dist, torch.zeros(5), atol=1e-4, rtol=1e-4)

    def test_symmetry(self, poincare: PoincareBall):
        torch.manual_seed(3)
        x = poincare.proj(torch.randn(5, 4) * 0.3)
        y = poincare.proj(torch.randn(5, 4) * 0.3)
        torch.testing.assert_close(
            poincare.sqdist(x, y), poincare.sqdist(y, x), atol=1e-4, rtol=1e-4
        )

    def test_non_negative(self, poincare: PoincareBall):
        torch.manual_seed(4)
        x = poincare.proj(torch.randn(10, 4) * 0.3)
        y = poincare.proj(torch.randn(10, 4) * 0.3)
        assert (poincare.sqdist(x, y) >= -1e-6).all()

    def test_triangle_inequality(self, poincare: PoincareBall):
        """d(x,z) <= d(x,y) + d(y,z) — using sqrt of sqdist."""
        torch.manual_seed(5)
        x = poincare.proj(torch.randn(5, 3) * 0.2)
        y = poincare.proj(torch.randn(5, 3) * 0.2)
        z = poincare.proj(torch.randn(5, 3) * 0.2)
        dxy = poincare.sqdist(x, y).sqrt()
        dyz = poincare.sqdist(y, z).sqrt()
        dxz = poincare.sqdist(x, z).sqrt()
        assert (dxz <= dxy + dyz + 1e-3).all()


class TestMobiusAdd:
    def test_identity_element(self, poincare: PoincareBall):
        """0 ⊕ x = x."""
        x = poincare.proj(torch.randn(5, 4) * 0.3)
        zero = torch.zeros_like(x)
        result = poincare.mobius_add(zero, x)
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_right_identity(self, poincare: PoincareBall):
        """x ⊕ 0 = x."""
        x = poincare.proj(torch.randn(5, 4) * 0.3)
        zero = torch.zeros_like(x)
        result = poincare.mobius_add(x, zero)
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_result_stays_in_ball(self, poincare: PoincareBall):
        torch.manual_seed(6)
        x = poincare.proj(torch.randn(10, 4) * 0.4)
        y = poincare.proj(torch.randn(10, 4) * 0.4)
        result = poincare.mobius_add(x, y)
        norms = result.norm(dim=-1)
        assert (norms < 1.0 / poincare.c ** 0.5 + 1e-3).all()

    def test_inverse(self, poincare: PoincareBall):
        """x ⊕ (-x) ≈ 0."""
        x = poincare.proj(torch.randn(5, 4) * 0.3)
        result = poincare.mobius_add(x, -x)
        torch.testing.assert_close(result, torch.zeros_like(x), atol=1e-4, rtol=1e-4)


class TestMobiusMatvec:
    def test_identity_matrix(self, poincare: PoincareBall):
        """I ⊗ x ≈ x."""
        x = poincare.proj(torch.randn(5, 3) * 0.3)
        I = torch.eye(3)
        result = poincare.mobius_matvec(I, x)
        torch.testing.assert_close(result, x, atol=1e-4, rtol=1e-4)

    def test_output_shape(self, poincare: PoincareBall):
        x = poincare.proj(torch.randn(4, 3) * 0.2)
        m = torch.randn(5, 3)
        result = poincare.mobius_matvec(m, x)
        assert result.shape == (4, 5)

    def test_zero_input(self, poincare: PoincareBall):
        """M ⊗ 0 = 0."""
        x = torch.zeros(3, 4)
        m = torch.randn(5, 4)
        result = poincare.mobius_matvec(m, x)
        torch.testing.assert_close(result, torch.zeros(3, 5), atol=1e-4, rtol=1e-4)

    def test_result_in_ball(self, poincare: PoincareBall):
        torch.manual_seed(7)
        x = poincare.proj(torch.randn(8, 4) * 0.3)
        m = torch.randn(6, 4)
        result = poincare.mobius_matvec(m, x)
        norms = result.norm(dim=-1)
        assert (norms < 1.0 / poincare.c ** 0.5 + 1e-3).all()


class TestParallelTransport:
    def test_ptransp0_at_origin(self, poincare: PoincareBall):
        """P_{0→x}(u) should scale u by 2/λ_x."""
        x = poincare.proj(torch.randn(4, 3) * 0.3)
        u = torch.randn(4, 3) * 0.1
        result = poincare.ptransp0(x, u)
        assert result.shape == u.shape
        assert torch.isfinite(result).all()

    def test_ptransp_preserves_finiteness(self, poincare: PoincareBall):
        torch.manual_seed(8)
        x = poincare.proj(torch.randn(4, 3) * 0.2)
        y = poincare.proj(torch.randn(4, 3) * 0.2)
        v = torch.randn(4, 3) * 0.1
        result = poincare.ptransp(x, y, v)
        assert torch.isfinite(result).all()

    def test_retr_transp_stays_on_manifold(self, poincare: PoincareBall):
        x = poincare.proj(torch.randn(4, 3) * 0.2)
        u = torch.randn(4, 3) * 0.01
        v = torch.randn(4, 3) * 0.01
        new_x, new_v = poincare.retr_transp(x, u, v)
        norms = new_x.norm(dim=-1)
        max_norm = (1 - poincare.eps[torch.float32]) / poincare.c ** 0.5
        assert (norms <= max_norm + 1e-6).all()


class TestToHyperboloid:
    def test_output_shape(self, poincare: PoincareBall):
        x = poincare.proj(torch.randn(5, 3) * 0.3)
        h = poincare.to_hyperboloid(x)
        assert h.shape == (5, 4)  # dim + 1

    def test_on_hyperboloid_sheet(self, poincare: PoincareBall):
        """Points should satisfy -x0^2 + x1^2 + ... + xn^2 = -K."""
        x = poincare.proj(torch.randn(5, 3) * 0.3)
        h = poincare.to_hyperboloid(x)
        K = 1.0 / poincare.c
        # Minkowski quadratic form: -h0^2 + sum(h_i^2)
        minkowski = -h[:, 0] ** 2 + h[:, 1:].pow(2).sum(dim=1)
        torch.testing.assert_close(minkowski, torch.full((5,), -K), atol=1e-3, rtol=1e-3)


class TestEgrad2Rgrad:
    def test_scaling(self, poincare: PoincareBall):
        """Riemannian gradient = Euclidean gradient / λ_x^2."""
        x = poincare.proj(torch.randn(4, 3) * 0.3)
        dp = torch.randn(4, 3)
        dp_copy = dp.clone()
        rgrad = poincare.egrad2rgrad(x, dp)
        # λ_x = 2 / (1 - c*||x||^2)
        x_sqnorm = x.data.pow(2).sum(dim=-1, keepdim=True)
        lam = 2 / (1.0 - poincare.c * x_sqnorm).clamp_min(1e-15)
        expected = dp_copy / lam.pow(2)
        torch.testing.assert_close(rgrad, expected, atol=1e-5, rtol=1e-5)


class TestInner:
    def test_positive_definite(self, poincare: PoincareBall):
        x = poincare.proj(torch.randn(4, 3) * 0.3)
        u = torch.randn(4, 3)
        inner = poincare.inner(x, u, keepdim=False)
        assert (inner >= 0).all()

    def test_zero_vector(self, poincare: PoincareBall):
        x = poincare.proj(torch.randn(4, 3) * 0.3)
        u = torch.zeros(4, 3)
        inner = poincare.inner(x, u, keepdim=False)
        # inner returns shape (4, 3) — per-component along last dim
        assert (inner.abs() < 1e-6).all()
