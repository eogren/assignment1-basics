import argparse
import logging
import pathlib
import threading
import time
from queue import Queue

from tqdm import tqdm

import wandb
from cs336_basics import bpe_token
from cs336_basics.tokenizer import serialize_merges, serialize_vocab


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_tokenizer",
        description="Train BPE Tokenizer",
    )

    parser.add_argument("-n", "--vocab-size", default=10000, type=int)
    parser.add_argument("-v", "--vocab-file", required=True)
    parser.add_argument("-m", "--merge-file", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("corpus")

    return parser


def run_tokenize(
    result_queue: Queue,
    corpus: str,
    vocab_size: int,
    special_tokens: list[str],
    handler: bpe_token.ProgressHandler,
):
    ret = bpe_token.tokenize(corpus, vocab_size, special_tokens, handler)
    result_queue.put(ret)


def main():
    parser = arg_parser()
    parsed = parser.parse_args()
    level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(level=level)

    corpus_path = pathlib.Path(parsed.corpus)
    if not corpus_path.exists():
        raise ValueError(f"{parsed.corpus} does not exist")

    queue = Queue(maxsize=1)
    special_tokens = ["<|endoftext|>"]

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="eogren-org",
        # Set the wandb project where this run will be logged.
        project="CS336",
        config={
            "corpus": parsed.corpus,
            "special_tokens": special_tokens,
            "vocab_size": parsed.vocab_size,
        },
        name=f"{corpus_path.name}-{parsed.vocab_size}",
        job_type="train_tokenizer",
        tags=[corpus_path.name],
    )

    handler = bpe_token.ProgressHandler()
    t = threading.Thread(
        name="training",
        target=run_tokenize,
        args=(queue, parsed.corpus, parsed.vocab_size, special_tokens, handler),
        daemon=True,
    )
    t.start()

    preshard_done = False
    preshard_pbar = None
    last_pbar = 0

    merge_pbar = None
    last_merge = 0

    while True:
        if not t.is_alive():
            if preshard_pbar is not None:
                preshard_pbar.close()

            if merge_pbar is not None:
                merge_pbar.close()

            break

        values = handler.values()

        if not preshard_done:
            if values.pretoken_total_shards != 0:
                if not preshard_pbar:
                    preshard_pbar = tqdm(total=values.pretoken_total_shards, desc="Pretokenizing")

                preshard_pbar.update(values.pretoken_done_shards - last_pbar)
                last_pbar = values.pretoken_done_shards

                if values.pretoken_done_shards == values.pretoken_total_shards:
                    preshard_done = True
                    preshard_pbar.close()
            time.sleep(0.1)
            continue

        if not merge_pbar:
            merge_pbar = tqdm(total=parsed.vocab_size, desc="Merging pairs")

        merge_pbar.update(values.tokenizer_merges_done - last_merge)
        last_merge = values.tokenizer_merges_done

        time.sleep(0.1)

    t.join()
    (vocab, merges) = queue.get(timeout=1)

    with open(parsed.vocab_file, "w") as v:
        serialize_vocab(v, vocab)

    with open(parsed.merge_file, "w") as m:
        serialize_merges(m, merges)

    largest_token_bytes = 0
    largest_token_string = ""

    for v in vocab.values():
        if len(v) > largest_token_bytes:
            largest_token_bytes = len(v)

        decoded = v.decode("utf-8", errors="replace")
        if len(decoded) > len(largest_token_string):
            largest_token_string = decoded

    artifact = wandb.Artifact(name="tokenizer", type="tokenizer")
    artifact.add_file(local_path=parsed.vocab_file, name="vocab")
    artifact.add_file(local_path=parsed.merge_file, name="merges")
    run.log_artifact(artifact)
    run.summary["largest_token_bytes"] = largest_token_bytes
    run.summary["largest_token_string"] = largest_token_string
    run.finish()
    print(f"Done! Serialized vocab to {parsed.vocab_file} and merges to {parsed.merge_file}")
