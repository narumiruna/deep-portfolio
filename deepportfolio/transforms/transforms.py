from typing import List

import torch

from . import functional as F


class Normalize(object):

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, self.mean, self.std)
