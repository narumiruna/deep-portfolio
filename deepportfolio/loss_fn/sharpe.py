import math

import mlconfig
import torch
from torch import nn


@mlconfig.register
class SharpeLoss(nn.Module):

    def __init__(self, unbiased: bool = False, annualized: bool = False, trading_days: int = 252):
        super().__init__()
        self.unbiased = unbiased
        self.annualized = annualized
        self.trading_days = trading_days

    def forward(self, out: torch.Tensor, y: torch.Tensor):
        returns = out.mul(y).sum(dim=2)

        sharpe = returns.mean(dim=1).div(returns.std(dim=1, unbiased=self.unbiased))

        if self.annualized:
            sharpe.mul_(math.sqrt(self.trading_days))

        return -sharpe.mean()
