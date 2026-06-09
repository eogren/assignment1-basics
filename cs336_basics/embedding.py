import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters"""
        super().__init__()

        self.lookup = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.lookup, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Take token_ids of shape (batch_size, sequence_length)"""
        return self.lookup[token_ids]


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None):
        """Construct an embedding module. This function should accept the following parameters:
        theta: Constant for RoPE
        d_k: dimension of query and key vectors
        max_seq_len: Max sequence length that will be input
        device: torch.device | None = None  Device to store the parameters on"""
        super().__init__()

        i = torch.arange(max_seq_len).reshape(max_seq_len, 1)
        k = (2 * torch.arange(start=1, end=1 + (d_k // 2)) - 2) / d_k
        theta_scalar = torch.tensor(theta)
        theta_tensor = torch.pow(theta_scalar, k)
        thetas = i / theta_tensor

        sines = torch.sin(thetas)
        cosines = torch.cos(thetas)

        self.register_buffer("sines", sines, persistent=False)
        self.register_buffer("cosines", cosines, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_evens = x[..., 0::2]
        x_odds = x[..., 1::2]

        x_rot_even = x_evens * self.cosines[token_positions] - x_odds * self.sines[token_positions]
        x_rot_odd = x_evens * self.sines[token_positions] + x_odds * self.cosines[token_positions]

        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd

        return out
