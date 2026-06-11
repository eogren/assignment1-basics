import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.activations import Swiglu
from cs336_basics.attention import MultiheadSelfAttention
from cs336_basics.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.ffn = Swiglu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, " ... seq_len d_model"]) -> Float[torch.Tensor, " ... seq_len d_model"]:
        first_chunk = x + self.attn(self.ln1(x))
        return first_chunk + self.ffn(self.ln2(first_chunk))
