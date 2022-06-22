import math

import mlconfig
import torch
from torch import nn


@mlconfig.register
class SortinoLoss(nn.Module):

    def __init__(self,
                 unbiased: bool = False,
                 log_return: bool = False,
                 annualized: bool = False,
                 trading_days: int = 252) -> None:
        super().__init__()
        self.unbiased = unbiased
        self.log_return = log_return
        self.annualized = annualized
        self.trading_days = trading_days

    def forward(self, out: torch.Tensor, y: torch.Tensor):
        returns = out.mul(y).sum(dim=2)

        if self.log_return:
            returns.add(1).log_()

        mean = returns.mean(dim=1)
        downside_risk = torch.masked_select(returns, returns.le(0)).std(unbiased=self.unbiased)
        sharpe = mean.div(downside_risk)

        if self.annualized:
            sharpe.mul(math.sqrt(self.trading_days))

        return -sharpe.mean()
