import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.activations import Swiglu
from cs336_basics.attention import MultiheadSelfAttention
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
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


class TransformerLM(nn.Module):
    layers: nn.ModuleList
    token_embeddings: Embedding
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        num_transformer_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if num_transformer_layers <= 0:
            raise ValueError("Transformer layers must be > 0")

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _i in range(num_transformer_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        tokens: Int[torch.Tensor, "... batch_size sequence_length"],
    ) -> Float[torch.Tensor, "... batch_size sequence_length vocab_size"]:
        intermediate = self.token_embeddings(tokens)
        for block in self.layers:
            intermediate = block(intermediate)

        intermediate = self.ln_final(intermediate)
        intermediate = self.lm_head(intermediate)
        # intermediate = softmax(intermediate, dim=-1)

        return intermediate
