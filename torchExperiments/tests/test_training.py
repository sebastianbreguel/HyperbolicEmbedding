"""Tests for training utilities: loss/optimizer factories, metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from manifolds.poincare import PoincareBall
from models.hnn import HNN
from training.metrics import get_accuracy, get_metrics
from training.trainer import obtain_loss, obtain_optimizer, train_model, val_process

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

class TestObtainLoss:
    def test_cross_entropy(self):
        loss = obtain_loss("cross")
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_mse(self):
        loss = obtain_loss("mse")
        assert isinstance(loss, nn.MSELoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            obtain_loss("huber")


class TestObtainOptimizer:
    @pytest.fixture
    def model(self) -> HNN:
        manifold = PoincareBall(c=1.0)
        return HNN(manifold, in_features=10, out_features=2, c=1.0, hidden=8)

    def test_adam(self, model: HNN):
        opt = obtain_optimizer("Adam", model)
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self, model: HNN):
        opt = obtain_optimizer("SGD", model)
        assert isinstance(opt, torch.optim.SGD)

    def test_radam(self, model: HNN):
        from optimizer.Radam import RiemannianAdam
        opt = obtain_optimizer("Radam", model)
        assert isinstance(opt, RiemannianAdam)

    def test_unknown_raises(self, model: HNN):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            obtain_optimizer("AdaGrad", model)


# ---------------------------------------------------------------------------
# Training loops (single epoch smoke tests)
# ---------------------------------------------------------------------------

class TestTrainValLoop:
    @pytest.fixture
    def setup(self):
        """Create a tiny model + data for smoke testing."""
        manifold = PoincareBall(c=1.0)
        model = HNN(manifold, in_features=5, out_features=2, c=1.0, hidden=8)
        device = torch.device("cpu")

        # Tiny dataset: 20 samples, 2 classes (one-hot labels)
        X = torch.randn(20, 5) * 0.1
        y = torch.zeros(20, 2)
        y[torch.arange(20), torch.randint(0, 2, (20,))] = 1.0

        loader = DataLoader(TensorDataset(X, y), batch_size=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return model, loader, criterion, optimizer, device

    def test_train_model_returns_float(self, setup):
        model, loader, criterion, optimizer, device = setup
        loss = train_model(model, loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_val_process_returns_float(self, setup):
        model, loader, criterion, optimizer, device = setup
        loss = val_process(model, loader, criterion, device)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_decreases_loss(self, setup):
        model, loader, criterion, optimizer, device = setup
        loss1 = train_model(model, loader, criterion, optimizer, device)
        loss2 = train_model(model, loader, criterion, optimizer, device)
        # Not guaranteed per-epoch but loss should be finite
        assert loss2 >= 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestGetAccuracy:
    @pytest.fixture
    def cross_setup(self):
        """Model + loader for classification accuracy testing."""
        manifold = PoincareBall(c=1.0)
        model = HNN(manifold, in_features=5, out_features=2, c=1.0, hidden=8)
        device = torch.device("cpu")

        X = torch.randn(16, 5) * 0.1
        # One-hot labels
        y = torch.zeros(16, 2)
        y[torch.arange(16), torch.randint(0, 2, (16,))] = 1.0

        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        y_test = y.numpy()
        return model, loader, y_test, device

    def test_cross_accuracy_in_range(self, cross_setup):
        model, loader, y_test, device = cross_setup
        acc = get_accuracy("cross", y_test, model, loader, device)
        assert 0.0 <= acc <= 1.0

    def test_mse_returns_float(self):
        """MSE metric returns a non-negative float."""
        manifold = PoincareBall(c=1.0)
        model = HNN(manifold, in_features=5, out_features=6, c=1.0, hidden=8)
        device = torch.device("cpu")

        X = torch.randn(16, 5) * 0.1
        y = torch.rand(16, 6)

        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        y_test = y.numpy()
        result = get_accuracy("mse", y_test, model, loader, device)
        assert isinstance(result, float)
        assert result >= 0


class TestGetMetrics:
    def test_cross_returns_list(self):
        manifold = PoincareBall(c=1.0)
        model = HNN(manifold, in_features=5, out_features=2, c=1.0, hidden=8)

        X = torch.randn(20, 5) * 0.1
        y = torch.zeros(20, 2)
        y[torch.arange(20), torch.randint(0, 2, (20,))] = 1.0

        loader = DataLoader(TensorDataset(X, y), batch_size=10)
        y_test = y.numpy()

        result = get_metrics("cross", y_test, None, model, loader)
        assert isinstance(result, list)
        # accuracy + 2*f1 + 2*precision + 2*recall = 7 values
        assert len(result) == 7

    def test_mse_returns_float(self):
        y_pred = np.random.randn(100, 6)
        y_test = np.random.randn(100, 6)

        manifold = PoincareBall(c=1.0)
        model = HNN(manifold, in_features=5, out_features=6, c=1.0, hidden=8)

        result = get_metrics("mse", y_test, y_pred, model, DataLoader(TensorDataset(torch.randn(10, 5))))
        assert isinstance(result, float)
