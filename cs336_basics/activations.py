from .linear import Linear
import torch
import torch.nn as nn


def _round_nearest_64(n: float) -> int:
    return 64 * ((n // 64) + 1)


class Swiglu(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int | None, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """Construct a SwiGLU  module. This function should accept the following parameters:
        d_model (int): Dimensionality of the feedforward input and output.
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters"""
        super().__init__()
        if d_ff:
            self.dff = d_ff
        else:
            self.dff = _round_nearest_64((8.0 / 3.0) * d_model)

        self.w1 = Linear(in_features=d_model, out_features=self.dff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=self.dff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=self.dff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)

        return self.w2(silu * w3_x)
