import argparse
import hashlib
import json
import math
import random
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim

import wandb
from cs336_basics.checkpoints import load_checkpoint, save_checkpoint
from cs336_basics.funcs import cosine_lr_schedule, gradient_clipping, pick_best_device
from cs336_basics.loaders import get_batch, get_eval_batch
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.optimizers import AdamW
from cs336_basics.transformer import TransformerLM


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

    def hash(self) -> str:
        params_to_ignore = ["device"]
        params = vars(self).copy()
        params["dtype"] = str(params["dtype"])
        for ignore in params_to_ignore:
            del params[ignore]

        params_str = json.dumps(params, sort_keys=True).encode("utf-8")
        return hashlib.sha256(params_str).hexdigest()[:12]


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


def validate(
    validation_arr: npt.NDArray, model: TransformerLM, parameters: HyperParameters, batch_size_multiplier: int = 8
) -> float:
    """Validate the model against the validation data and return the average cross entropy loss."""
    running_loss: float = 0.0
    tokens: float = 0

    with torch.no_grad():
        for train, test in get_eval_batch(
            validation_arr, parameters.batch_size * batch_size_multiplier, parameters.context_length, parameters.device
        ):
            predictions = model(train)
            loss = cross_entropy_loss(predictions, test, reduction="sum")
            running_loss += loss.item()
            tokens += train.numel()

    return running_loss / tokens


def train(
    ds_name: str,
    input_path: str | PathLike,
    validation_path: str | PathLike,
    parameters: HyperParameters,
    checkpoint_info: CheckpointInfo | None = None,
    resume: bool = False,
):
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

    name = f"{ds_name}-{parameters.hash()}"
    lowest_loss: float | None = None

    # Resume from the latest checkpoint if requested and present. The run name is
    # deterministic from hyperparameters, so the same config always maps to the
    # same checkpoint file and the same wandb run -> idempotent under eviction.
    start_step = 0
    if resume and checkpoint_info is not None:
        ckpt_path = Path(checkpoint_info.save_directory) / f"{name}-latest.pt"
        if ckpt_path.exists():
            start_step = load_checkpoint(ckpt_path, model, optimizer) + 1
            print(f"Resuming {name} from step {start_step}")

    # Deterministic wandb id so an eviction continues the SAME run, not a new one.
    run_id = hashlib.sha1(name.encode()).hexdigest()[:16]
    with wandb.init(
        entity="eogren-org",
        project="CS336",
        config=wandb_config,
        job_type="train_llm",
        tags=["smoke-test"],
        name=name,
        id=run_id,
        resume="allow",
    ) as run:
        run.watch(model)
        for step in range(start_step, total_steps):
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
            if not torch.isfinite(loss):
                run.summary["diverged"] = True
                run.summary["diverged_step"] = step
                break
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

            run.log({"train/lr": new_lr, "train/loss": loss.item()}, step=step)

            if step > 0 and step % 200 == 0:
                validation_loss = validate(validation_arr, model, parameters, 8)
                run.log({"validation/loss": validation_loss}, step=step)

                if lowest_loss is None or validation_loss < lowest_loss:
                    lowest_loss = validation_loss
                    save_best_checkpoint(name, model, step, checkpoint_info)

            if checkpoint_info is not None and (step % checkpoint_info.save_interval == 0 or step == total_steps - 1):
                save_latest_checkpoint(name, model, optimizer, step, checkpoint_info)


def save_latest_checkpoint(
    name: str, model: TransformerLM, optimizer: torch.optim.Optimizer, step: int, checkpoint_info: CheckpointInfo
):
    out = Path(checkpoint_info.save_directory) / f"{name}-latest.pt"
    save_checkpoint(model, optimizer, step, out)


def save_best_checkpoint(name: str, model: TransformerLM, step: int, checkpoint_info: CheckpointInfo | None):
    if checkpoint_info:
        out = Path(checkpoint_info.save_directory)
        save_checkpoint(model, None, step, out / f"{name}-best.pt")


def parse_lr(s: Any) -> float:
    val = float(s)
    if val <= 0:
        raise argparse.ArgumentTypeError("LR must be > 0")

    return val


def argument_parser() -> ArgumentParser:
    ret = ArgumentParser()
    ret.add_argument("-c", "--checkpoint", required=False, help="Directory for checkpoints.")
    ret.add_argument("-ci", "--checkpoint_interval", required=False, type=int, default=200)
    ret.add_argument(
        "--lr",
        required=False,
        type=parse_lr,
        default=0.001,
        help="Learning rate for the training session",
    )
    ret.add_argument("--total-tokens", type=int, default=7000000, help="number of tokens to process")
    ret.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    ret.add_argument("--context-length", type=int, default=256, help="Context length for training")
    ret.add_argument("--ds-name", default="tinystories", help="Dataset name; used in run/checkpoint naming.")
    ret.add_argument(
        "--input",
        default="tokenized_datasets/tinystories_10k/train.np",
        help="Path to tokenized uint16 training memmap.",
    )
    ret.add_argument(
        "--valid",
        default="tokenized_datasets/tinystories_10k/valid.np",
        help="Path to tokenized uint16 validation memmap.",
    )
    ret.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Resume from {name}-latest.pt if it exists (default: on; use --no-resume to start fresh).",
    )
    return ret


def main():
    args = argument_parser().parse_args()
    config = HyperParameters(
        lr=args.lr,
        total_tokens_to_process=args.total_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    checkpoint_info = None
    if args.checkpoint is not None:
        checkpoint_info = CheckpointInfo(
            save_directory=args.checkpoint,
            save_interval=args.checkpoint_interval,
        )

    train(
        args.ds_name,
        args.input,
        args.valid,
        config,
        checkpoint_info,
        resume=args.resume,
    )


@dataclass
class CheckpointInfo:
    save_directory: str | PathLike
    save_interval: int


if __name__ == "__main__":
    main()
