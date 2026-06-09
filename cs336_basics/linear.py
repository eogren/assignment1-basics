import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module. This function should accept the following parameters:
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters"""
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        var = 2.0 / (in_features + out_features)
        stddev = math.sqrt(var)

        nn.init.trunc_normal_(self.weight, mean=0, std=stddev, a=-3.0 * stddev, b=3.0 * stddev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return torch.einsum("o i, ... i -> ... o", self.weight, x)
