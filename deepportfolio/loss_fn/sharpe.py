import mlconfig
import torch
from torch import nn


@mlconfig.register
class SharpeRatioLoss(nn.Module):

    def forward(self, out: torch.Tensor, y: torch.Tensor):
        returns = out.mul(y).sum(dim=2)
        sharpe = returns.mean(dim=1) / returns.std(dim=1, unbiased=True)
        return -sharpe.mean()
