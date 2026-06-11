import einops
import torch

from jaxtyping import Bool, Float


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension 𝑖, and apply softmax to the 𝑖-th dimension of the
    input tensor. The output tensor should have the same shape as the input tensor, but its 𝑖-th
    dimension will now have a normalized probability distribution. Use the trick of subtracting the
    maximum value in the 𝑖-th dimension from all elements of the 𝑖-th dimension to avoid numerical
    stability issues."""
    # Get max values across dim and subtract everything from it
    max_values = torch.max(x, dim=dim, keepdim=True).values
    stability_fixed = x - max_values

    # Now we can do e^x for everything in the tensor
    exponated = torch.exp(stability_fixed)

    # Now sum across dim (bottom of the softmax equation)
    sums = torch.sum(exponated, dim=dim, keepdim=True)

    # Now divide
    ret = exponated / sums

    return ret


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    pre_softmax = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5)

    if mask is not None:
        pre_softmax.masked_fill_(~mask, -torch.inf)
    post_softmax: Float[torch.Tensor, " ... queries keys"] = softmax(pre_softmax, -1)

    return einops.einsum(post_softmax, V, "... queries keys, ... keys d_v -> ... queries d_v")
