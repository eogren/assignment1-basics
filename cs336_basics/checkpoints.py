import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    to_save = {"model": model.state_dict(), "iteration": iteration}
    if optimizer is not None:
        to_save["optimizer"] = optimizer.state_dict()

    # Write to a temp file then atomically rename, so a crash/eviction mid-save
    # can never leave a torn checkpoint as the file we'd resume from (also makes
    # the async rclone upload safe -- it only ever sees a complete file).
    if isinstance(out, (str, os.PathLike)):
        out = os.fspath(out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)  # dir may not exist on a fresh checkout/pod
        tmp = f"{out}.tmp"
        torch.save(to_save, tmp)
        os.replace(tmp, out)  # atomic within the same filesystem
    else:
        torch.save(to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
