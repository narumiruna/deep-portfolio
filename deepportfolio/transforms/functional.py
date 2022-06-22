from typing import List

import torch


def normalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean: torch.Tensor = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std: torch.Tensor = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    if mean.ndim == 1:
        mean = mean.view(1, -1)
    if std.ndim == 1:
        std = std.view(1, -1)

    tensor.sub_(mean).div_(std)

    return tensor
