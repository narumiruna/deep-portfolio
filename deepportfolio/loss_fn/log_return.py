import mlconfig
import torch
from torch import nn


@mlconfig.register
class LogReturnLoss(nn.Module):

    def forward(self, out: torch.Tensor, y: torch.Tensor):
        returns = out.mul(y).sum(dim=2).add(1).log()
        return -returns.mean()
