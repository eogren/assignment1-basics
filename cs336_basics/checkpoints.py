import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    to_save = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}

    torch.save(to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
