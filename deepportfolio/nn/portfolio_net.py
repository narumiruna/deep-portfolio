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
                 rnn_type: str = 'LSTM'):
        super(PortfolioNet, self).__init__()
        if rnn_type not in ('RNN', 'GRU', 'LSTM'):
            raise ValueError('rnn_type should be RNN, GRU or LSTM')

        self.rnn = getattr(nn, rnn_type)(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        out, _ = self.rnn(x)  # (seq_len, batch_size, hidden_dim)

        out = out.view(-1, out.size(2))  # (seq_len x batch_size, hidden_dim)
        out = self.fc(out)  # (seq_len x batch_size, output_dim)

        out = out.view(seq_len, batch_size, -1)  # (seq_len, batch_size, output_dim)
        out = out.permute(1, 0, 2)  # (batch_size, seq_len, output_dim)

        out = F.softmax(out, dim=2)

        return out
