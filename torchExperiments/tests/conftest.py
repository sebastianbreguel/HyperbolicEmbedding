"""Shared fixtures for torchExperiments tests."""

from __future__ import annotations

import sys
import types

# Provide a stub for umap so that training/__init__.py -> data_gen.py can import
# without requiring pkg_resources at test time.
if "umap" not in sys.modules:
    sys.modules["umap"] = types.ModuleType("umap")

import pytest
import torch

from manifolds.euclidean import Euclidean
from manifolds.poincare import PoincareBall


@pytest.fixture
def poincare() -> PoincareBall:
    return PoincareBall(c=1.0)


@pytest.fixture
def euclidean() -> Euclidean:
    return Euclidean()


@pytest.fixture
def small_ball_points() -> torch.Tensor:
    """Points well inside the unit ball (norm < 0.5)."""
    torch.manual_seed(42)
    pts = torch.randn(8, 4) * 0.3
    return pts


@pytest.fixture
def origin_4d() -> torch.Tensor:
    return torch.zeros(1, 4)
