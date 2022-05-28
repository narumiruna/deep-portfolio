from torch import nn


class PortfolioNet(nn.Sequential):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 batch_first: bool = True,
                 rnn_type: str = 'LSTM'):
        super(PortfolioNet, self).__init__()
        if rnn_type not in ('RNN', 'GRU', 'LSTM'):
            raise ValueError('rnn_type should be RNN, GRU or LSTM')

        self.rnn = getattr(nn, rnn_type)(input_size=input_dim,
                                         hidden_size=hidden_dim,
                                         num_layers=num_layers,
                                         batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
