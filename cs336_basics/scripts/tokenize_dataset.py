import argparse
import itertools
import logging
import mmap
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics import Tokenizer


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_tokenizer",
        description="Train BPE Tokenizer",
    )

    parser.add_argument("-v", "--vocab-file", required=True)
    parser.add_argument("-m", "--merge-file", required=True)
    parser.add_argument("-o", "--output-file", required=True)
    parser.add_argument("-s", "--special-tokens", nargs="+", action="extend")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("corpus")

    return parser


def chunked_file(path: str | Path, special_tokens: str, pbar, ideal_chunk: int = 1024 * 1024) -> Iterable[bytes]:
    parsed_path = Path(path)
    size = parsed_path.stat().st_size
    with open(path, "r+b") as f:
        idx = 0
        special_tokens_bytes = special_tokens.encode("utf-8")
        mmaped = mmap.mmap(f.fileno(), length=0, prot=mmap.PROT_READ)

        while True:
            end = idx + ideal_chunk
            if end > size:
                pbar.update(end - idx)
                yield mmaped[idx:end]
                break

            delimiter = mmaped.find(special_tokens_bytes, end)
            end = delimiter if delimiter != -1 else size

            pbar.update(end - idx)
            yield mmaped[idx:end]
            idx = end


def main():
    parsed = arg_parser().parse_args()
    level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(level=level)

    special_tokens = parsed.special_tokens if parsed.special_tokens is not None else ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(parsed.vocab_file, parsed.merge_file, special_tokens)

    with open(parsed.output_file, "b+w") as output:
        input = Path(parsed.corpus)
        input_size = input.stat().st_size
        with tqdm(total=input_size, unit="B", unit_scale=True) as pbar:
            chunked = chunked_file(input, special_tokens[0], pbar)
            for tokens in itertools.batched(tokenizer.encode_iterable(chunked), n=64000000):
                np.array(tokens, dtype=np.uint16).tofile(output)
