import mlconfig
import torch
from torch import nn


@mlconfig.register
class Profit(object):
    """A meter to calculate portfolio profit"""

    def __init__(self):
        self.profit = 1.0

    def update(self, weights: torch.Tensor, returns: torch.Tensor):
        # assets, returns: batch_size, seq_len, num_assets

        portfolio_returns = weights.mul(returns).sum(dim=2)
        portfolio_returns = portfolio_returns[:, -1]

        self.profit *= portfolio_returns.add(1.0).prod().item()

    @property
    def value(self) -> float:
        return self.profit
