import torch
from deepportfolio import nn


@torch.no_grad()
def test_nn_portfolionet_forward():
    batch_size = 2
    input_dim = 3
    output_dim = 4
    hidden_dim = 5
    seq_len = 50

    rnn_type = 'LSTM'

    m = nn.PortfolioNet(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, rnn_type=rnn_type)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = m(x)

    assert list(y.size()) == [batch_size, seq_len, output_dim]
