import torch
import torch.nn.functional as F

from deepportfolio.loss_fn import LogReturnLoss
from deepportfolio.loss_fn import SharpeRatioLoss


@torch.no_grad()
def test_loss_fn_sharpe_ratio_loss_forward():
    batch_size = 2
    num_assets = 3
    seq_len = 4

    w = F.softmax(torch.randn(batch_size, seq_len, num_assets), dim=2)
    r = torch.randn(batch_size, seq_len, num_assets)

    fn = SharpeRatioLoss()
    l = fn(w, r)

    assert l.ndim == 0


@torch.no_grad()
def test_loss_fn_log_return_loss_forward():
    batch_size = 2
    num_assets = 3
    seq_len = 4

    w = F.softmax(torch.randn(batch_size, seq_len, num_assets), dim=2)
    r = torch.randn(batch_size, seq_len, num_assets)

    fn = LogReturnLoss()
    loss = fn(w, r)

    assert loss.ndim == 0
