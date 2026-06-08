import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Construct an RMS norm module. This function should accept the following parameters:
            d_model: Dimension of the hidden parameters
            eps: Epsilon value for numeric stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [... in_features]
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt((1.0/x.shape[-1]) * torch.sum(torch.square(x), dim=-1, keepdim=True) + self.eps)
        result = (x / rms_x) * self.weight

        assert x.shape == result.shape

        return result.to(in_dtype)