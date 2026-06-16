import math
import random
from os import PathLike
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import wandb
from cs336_basics.funcs import cosine_lr_schedule, gradient_clipping
from cs336_basics.loaders import get_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.optimizers import AdamW
from cs336_basics.transformer import TransformerLM


def pick_best_device() -> torch.device:
    """Pick best device for training - GPUs if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class HyperParameters:
    vocab_size: int
    batch_size: int
    context_length: int
    d_model: int
    d_ff: int
    rope_theta: float
    num_layers: int
    num_heads: int
    total_tokens_to_process: int
    device: torch.device
    dtype: torch.dtype

    # Optimizer
    lr: float
    lr_max: float
    lr_min: float
    weight_decay: float
    betas: tuple[float, float]
    optimizer_eps: float
    max_gradient_l2: float

    def __init__(
        self,
        vocab_size: int = 10000,
        batch_size: int = 64,
        context_length: int = 256,
        d_model: int = 512,
        d_ff: int = 1344,
        rope_theta: float = 10000.0,
        num_layers: int = 4,
        num_heads: int = 16,
        total_tokens_to_process: int = 327680000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 10**-8,
        max_gradient_l2: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.total_tokens_to_process = total_tokens_to_process
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.optimizer_eps = optimizer_eps
        self.max_gradient_l2 = max_gradient_l2
        self.lr_max = lr
        self.lr_min = lr / 10

        if device is None:
            device = pick_best_device()

        self.device = device

        if dtype is None:
            dtype = torch.float32

        self.dtype = dtype

    @property
    def warmup(self) -> int:
        return math.floor(float(self.target_steps) * 0.05)

    @property
    def target_steps(self) -> int:
        return math.ceil(self.total_tokens_to_process / (float(self.batch_size * self.context_length)))

    def to_dict(self) -> dict[str, Any]:
        props = vars(self).copy()
        props["dtype"] = str(self.dtype)
        props["device"] = str(self.device)
        props["warmup"] = self.warmup
        props["target_steps"] = self.target_steps

        return props


def build_model(parameters: HyperParameters) -> TransformerLM:
    return TransformerLM(
        vocab_size=parameters.vocab_size,
        num_transformer_layers=parameters.num_layers,
        d_model=parameters.d_model,
        num_heads=parameters.num_heads,
        d_ff=parameters.d_ff,
        max_seq_len=parameters.context_length,
        theta=parameters.rope_theta,
        device=parameters.device,
        dtype=parameters.dtype,
    )


def build_optimizer(parameters: HyperParameters, model: nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=parameters.lr,
        weight_decay=parameters.weight_decay,
        betas=parameters.betas,
        eps=parameters.optimizer_eps,
    )


def train(input_path: str | PathLike, validation_path: str | PathLike, parameters: HyperParameters):
    input_arr = np.memmap(input_path, dtype=np.uint16)
    validation_arr = np.memmap(validation_path, dtype=np.uint16)

    total_steps = parameters.target_steps
    model = build_model(parameters)
    optimizer = build_optimizer(parameters, model)
    print(model)
    print(optimizer)
    print(f"Would do {total_steps} steps")

    seed = 42
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)

    wandb_config = parameters.to_dict()
    wandb_config["rng_seed"] = seed

    with wandb.init(
        entity="eogren-org", project="CS336", config=wandb_config, job_type="train_llm", tags=["smoke-test"]
    ) as run:
        run.watch(model)
        for step in range(total_steps):
            model.train()

            (training_data, labels) = get_batch(
                dataset=input_arr,
                batch_size=parameters.batch_size,
                context_length=parameters.context_length,
                device=parameters.device,
                rng=rng,
            )

            optimizer.zero_grad()
            predictions = model(training_data)
            loss = cross_entropy_loss(predictions, labels)
            loss.backward()
            gradient_clipping(model.parameters(), parameters.max_gradient_l2)
            new_lr = cosine_lr_schedule(
                it=step,
                max_learning_rate=parameters.lr_max,
                min_learning_rate=parameters.lr_min,
                warmup_iters=parameters.warmup,
                cosine_cycle_iters=total_steps,
            )
            for g in optimizer.param_groups:
                g["lr"] = new_lr

            optimizer.step()

            run.log({"train/lr": new_lr, "train/loss": loss.item(), "train/step": step}, step=step)


def main():
    config = HyperParameters()
    config.total_tokens_to_process = 2000 * config.batch_size * config.context_length
    assert config.target_steps == 2000

    train("tokenized_datasets/tinystories_10k/train.np", "tokenized_datasets/tinystories_10k/valid.np", config)


if __name__ == "__main__":
    main()
