import argparse
import logging
import threading
import time

from cs336_basics import bpe_token
from cs336_basics.tokenizer import serialize_merges, serialize_vocab
from tqdm import tqdm
from queue import Queue


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

    queue = Queue(maxsize=1)

    handler = bpe_token.ProgressHandler()
    t = threading.Thread(
        name="training",
        target=run_tokenize,
        args=(queue, parsed.corpus, parsed.vocab_size, ["<|endoftext|>"], handler),
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

    print("Done! Serialized vocab to {} and merges to {}", parsed.vocab_file, parsed.merge_file)
