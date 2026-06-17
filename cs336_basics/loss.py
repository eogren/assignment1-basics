from typing import Literal

import torch
from jaxtyping import Float, Int


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... batch_size vocab_size"],
    targets: Int[torch.Tensor, "... batch_size"],
    reduction: Literal["sum", "mean"] = "mean",
) -> Float[torch.Tensor, ""]:
    max_values = torch.max(logits, dim=-1, keepdim=True).values
    stability_fixed = logits - max_values
    targets_unsqueezed = targets.unsqueeze(-1)
    gathered = torch.gather(logits, dim=-1, index=targets_unsqueezed)

    exponated = torch.exp(stability_fixed)
    summed = torch.sum(exponated, dim=-1, keepdim=True)
    logged = torch.log(summed)

    if reduction == "mean":
        return torch.mean(-gathered + max_values + logged)
    elif reduction == "sum":
        return torch.sum(-gathered + max_values + logged)
    else:
        raise ValueError(f"Unexpected reduction type {reduction}")


if __name__ == "__main__":
    sample_logit = torch.tensor([[1, 2, 3], [5, 9, 11]])

    sample_targets = torch.tensor([2, 1])

    print(cross_entropy_loss(sample_logit, sample_targets))
