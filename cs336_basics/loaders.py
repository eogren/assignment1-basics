from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch


def get_eval_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str | torch.device | None = None
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    _validate_inputs(dataset, context_length)
    current_batch = 0
    max_batch = (dataset.shape[0] - 1) // context_length

    while True:
        end_batch = min(current_batch + batch_size, max_batch)
        start_indices = np.arange(current_batch * context_length, end_batch * context_length, step=context_length)
        yield _slice_dataset(dataset, start_indices, context_length, device)

        if end_batch == max_batch:
            break

        current_batch = end_batch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.Tensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    _validate_inputs(dataset, context_length)
    # We always need context_length + 1 tokens to get the label sequence
    max_index = dataset.shape[0] - context_length

    if rng is None:
        rng = np.random.default_rng()

    start_indices = rng.integers(low=0, high=max_index, size=batch_size)
    return _slice_dataset(dataset, start_indices, context_length, device)


def _validate_inputs(dataset: npt.NDArray, context_length: int):
    if dataset.ndim != 1:
        raise ValueError("dataset must be 1D")

    if dataset.shape[0] < context_length + 1:
        raise ValueError(f"can't sample enough entries - need {context_length + 1} but have {dataset.shape[0]}")


def _slice_dataset(
    dataset: npt.NDArray, start_indices: npt.NDArray, context_length: int, device: torch.device | str | None
):
    index_offsets = np.arange(0, context_length)
    grid = start_indices[:, np.newaxis] + index_offsets

    samples = dataset[grid].astype(np.int64)
    targets = dataset[grid + 1].astype(np.int64)

    ret = (torch.tensor(samples, device=device), torch.tensor(targets, device=device))

    assert ret[0].dtype == torch.int64
    assert ret[1].dtype == torch.int64
    return ret
