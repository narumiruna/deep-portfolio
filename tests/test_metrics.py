import pytest
import torch

from deepportfolio.metrics import Profit


def test_metrics_profit():
    m = Profit()
    weights = torch.tensor([[[0.6, 0.4]]])  # assets
    returns = torch.tensor([[[0.2, -0.2]]])  # returns
    m.update(weights, returns)

    assert m.value == pytest.approx(1.04)
