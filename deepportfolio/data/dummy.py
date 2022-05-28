from typing import Tuple

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __init__(self, seq_len: int = 50, input_dim: int = 8, output_dim: int = 4, num_samples: int = 100):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.seq_len, self.input_dim)
        y = torch.randn(self.output_dim)
        return x, y

    def __len__(self) -> int:
        return self.num_samples
