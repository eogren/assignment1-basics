import einops
import torch
import torch.nn as nn
from cs336_basics.embedding import RoPE
import cs336_basics.funcs

from cs336_basics import Linear
from jaxtyping import Float


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Implement causal multi-head self-attention as a torch.nn.Module. Your
        implementation should accept (at least) the following parameters:

            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            max_seq_len: int Max sequence length that will be passed in"""
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        d_v = d_k
        assert d_k % 2 == 0

        self.num_heads = num_heads
        self.d_k = d_k

        self.w_q = Linear(out_features=num_heads * d_k, in_features=d_model, device=device, dtype=dtype)
        self.w_k = Linear(out_features=num_heads * d_k, in_features=d_model, device=device, dtype=dtype)
        self.w_v = Linear(out_features=num_heads * d_v, in_features=d_model, device=device, dtype=dtype)
        self.w_o = Linear(out_features=d_model, in_features=num_heads * d_v, device=device, dtype=dtype)
        self.rope = RoPE(theta=10000, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
        self, x: Float[torch.Tensor, " ... sequence_length d_model"]
    ) -> Float[torch.Tensor, " ... sequence_length d_model"]:
        seq_len = x.shape[-2]
        print(f"seq_len is {seq_len}")
        """Take token_ids of shape (batch_size, sequence_length)"""
        print(f"x is shape {x.shape}")
        proj_q = self.w_q(x)
        print(f"proj_q is shape {proj_q.shape}")
        proj_k = self.w_k(x)
        proj_v = self.w_v(x)

        # Reshape for heads
        proj_q = einops.rearrange(
            proj_q, "... sequence_length (num_heads dk) -> ... sequence_length num_heads dk", num_heads=self.num_heads
        )
        proj_k = einops.rearrange(
            proj_k, "... sequence_length (num_heads dk) -> ... sequence_length num_heads dk", num_heads=self.num_heads
        )

        # Apply RoPE
        token_positions = torch.arange(0, seq_len, dtype=torch.int, device=proj_q.device)
        token_positions = token_positions.unsqueeze(0)
        print(f"proj_q before rope: {proj_q}")
        proj_q = self.rope(proj_q, token_positions)
        print(f"proj_q after rope: {proj_q}")
        proj_k = self.rope(proj_k, token_positions)

        # Generate mask

        # Them multihead

        return torch.empty()
