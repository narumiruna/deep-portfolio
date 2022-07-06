import mlconfig
import torch
import torch.nn.functional as F
from torch import nn


@mlconfig.register
class PortfolioNet(nn.Sequential):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 rnn_type: str = 'LSTM',
                 short_selling: bool = False,
                 leverage: float = 1.0):
        super(PortfolioNet, self).__init__()
        self.short_selling = short_selling
        self.leverage = leverage

        if rnn_type not in ('RNN', 'GRU', 'LSTM'):
            raise ValueError('rnn_type should be RNN, GRU or LSTM')

        if leverage <= 0:
            raise ValueError('leverage should be greater than 0')

        self.rnn = getattr(nn, rnn_type)(input_size=input_dim,
                                         hidden_size=hidden_dim,
                                         num_layers=num_layers,
                                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim)
        out = self.fc(out)  # (batch_size, seq_len, output_dim)

        if self.short_selling:
            out = out.sign() * F.softmax(out.abs(), dim=2)
        else:
            out = F.softmax(out, dim=2)

        out.mul_(self.leverage)

        return out
