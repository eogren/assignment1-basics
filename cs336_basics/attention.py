import einops
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int

from cs336_basics.embedding import RoPE
from cs336_basics.funcs import scaled_dot_product_attention
from cs336_basics.linear import Linear


def _triangle_mask(seq_len: int, device: torch.device | None = None) -> Bool[torch.Tensor, "seq_len seq_len"]:
    mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    return torch.tril(mask)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,  # pass in max_seq_len to use RoPE
        theta: float | None = None,  # pass in theta to use RoPE
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
        self.device = device

        self.q_proj = Linear(out_features=num_heads * d_k, in_features=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(out_features=num_heads * d_k, in_features=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(out_features=num_heads * d_v, in_features=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(out_features=d_model, in_features=num_heads * d_v, device=device, dtype=dtype)
        if theta:
            assert max_seq_len is not None, "max_seq_len must be set if theta is set"
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, " ... sequence_length d_model"]:
        seq_len = x.shape[-2]

        proj_q = self.q_proj(x)
        proj_k = self.k_proj(x)
        proj_v = self.v_proj(x)

        # Reshape for heads
        proj_q = einops.rearrange(
            proj_q, "... sequence_length (num_heads dk) -> ... num_heads sequence_length dk", num_heads=self.num_heads
        )
        # ok now we need (batch_id x num_heads x sequence_length x dk)
        proj_k = einops.rearrange(
            proj_k, "... sequence_length (num_heads dk) -> ... num_heads sequence_length dk", num_heads=self.num_heads
        )

        proj_v = einops.rearrange(
            proj_v, "... sequence_length (num_heads dk) -> ... num_heads sequence_length dk", num_heads=self.num_heads
        )

        if self.rope:
            if token_positions is None:
                # Apply RoPE
                token_positions = torch.arange(0, seq_len, dtype=torch.int, device=proj_q.device)
                token_positions = token_positions.unsqueeze(0)
            proj_q = self.rope(proj_q, token_positions)
            proj_k = self.rope(proj_k, token_positions)

        # Generate mask - should be seq_len by seq_len
        mask = _triangle_mask(seq_len, device=self.device)

        # Them multihead
        multihead = scaled_dot_product_attention(proj_q, proj_k, proj_v, mask)

        # now reshape again so we are concatting things together
        multihead = einops.rearrange(
            multihead, "... num_heads sequence_length dk -> ... sequence_length (num_heads dk)"
        )

        ret = self.output_proj(multihead)
        return ret
