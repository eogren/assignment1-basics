from __future__ import annotations

from collections import Counter
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite
from collections.abc import Iterable, Iterator


_SAFE_RANGES = [range(0x21, 0x7F), range(0xA1, 0xAD), range(0xAE, 0x100)]


def _build_byte_map() -> list[str]:
    """Build a serializable map of byte (0-255) -> printable char."""
    ret = []
    next_char_idx = 256

    for b in range(0, 256):
        if _is_printable(b):
            ret.append(chr(b))
        else:
            ret.append(chr(next_char_idx))
            next_char_idx = next_char_idx + 1

    return ret


def _is_printable(b: int) -> bool:
    """Check whether the given byte is printable. Must be between 0 and 255."""
    if b < 0 or b > 255:
        raise ValueError("b must be 0-255")

    return any(b in range for range in _SAFE_RANGES)


_BYTE_MAP = _build_byte_map()
_STR_TO_BYTE_MAP: dict[str, int] = {_BYTE_MAP[i]: i for i in range(0, len(_BYTE_MAP))}

_CURRENT_VOCAB_VER = 1


def _bytes_to_printable(bytes_in: bytes) -> str:
    """Use the byte map to convert the set of bytes into a printable string."""
    chars = [_BYTE_MAP[b] for b in bytes_in]
    return str.join("", chars)


def _printable_to_bytes(str_in: str) -> bytes:
    """Convert a encoded printable string back to into its original bytes"""
    return bytes([_STR_TO_BYTE_MAP[s] for s in str_in])


def serialize_vocab(out: SupportsWrite[str], vocab: dict[int, bytes]):
    """Serialize the given vocab to file path, overwriting it if it exists."""
    printable_vocab = {k: _bytes_to_printable(v) for k, v in vocab.items()}

    json.dump(dict(version=_CURRENT_VOCAB_VER, vocab=printable_vocab), out)


def deserialize_vocab(input: SupportsRead[str]) -> dict[int, bytes]:
    parsed = json.load(input)
    if "version" not in parsed or parsed["version"] != _CURRENT_VOCAB_VER:
        raise ValueError(f"Failed to parse object {0} - version not found or wrong", parsed)

    ret = dict()

    for k, v in parsed["vocab"].items():
        ret[int(k)] = _printable_to_bytes(v)

    return ret


def serialize_merges(out: SupportsWrite[str], merges: list[tuple[bytes, bytes]]):
    """Serialize the given merges to the output buffer."""
    printable_merges = [
        dict(first=_bytes_to_printable(first), second=_bytes_to_printable(second)) for first, second in merges
    ]

    json.dump(dict(version=_CURRENT_VOCAB_VER, merges=printable_merges), out)


def deserialize_merge(input: SupportsRead[str]) -> list[tuple[bytes, bytes]]:
    """Deserialize merges serialized with serialize_merges."""
    parsed = json.load(input)
    if "version" not in parsed or parsed["version"] != _CURRENT_VOCAB_VER:
        raise ValueError(f"Failed to parse object {0} - version not found or wrong", parsed)

    ret = []

    for obj in parsed["merges"]:
        first = _printable_to_bytes(obj["first"])
        second = _printable_to_bytes(obj["second"])

        ret.append((first, second))

    return ret


def _validate_vocab(vocab: dict[int, bytes]):
    """Check whether the vocab is valid. No bytes should appear twice in value."""
    cnt = Counter()
    for b in vocab.values():
        cnt[b] += 1

    most_common = cnt.most_common(n=1)[0]
    if most_common[1] != 1:
        raise ValueError("Vocab seems corrupt - {} appears more than once in values", most_common)


def _validate_merges(merges: list[tuple[bytes, bytes]]):
    cnt = Counter()
    for first, second in merges:
        cnt[first] += 1

    most_common = cnt.most_common(n=1)[0]
    if most_common[1] != 1:
        raise ValueError("Merges seems corrupt - {} appears more than once as merge source", most_common)


def _convert_merges_to_tokens(
    reverse_vocab: dict[bytes, int], merges: list[tuple[bytes, bytes]]
) -> list[tuple[int, int, int]]:
    """We get merges as character pairs like "b", "e". But for the tokenizer it's easier if we get (2, 3) -> 5 type merges."""
    token_merges = []
    for merge in merges:
        first_token = reverse_vocab[merge[0]]
        second_token = reverse_vocab[merge[1]]
        new_token = reverse_vocab[merge[0] + merge[1]]

        token_merges.append((first_token, second_token, new_token))

    return token_merges


class Tokenizer:
    _vocab: dict[int, bytes]
    _merges: list[tuple[bytes, bytes]]
    _token_merges: list[tuple[int, int, int]]
    _special_tokens: list[str]

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        _validate_vocab(vocab)
        self._vocab = vocab

        _validate_merges(merges)
        self._merges = merges

        reverse_vocab = {v: k for k, v in vocab.items()}
        self._token_merges = _convert_merges_to_tokens(reverse_vocab, merges)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = None
        merges = None

        with open(vocab_filepath) as f:
            vocab = deserialize_vocab(f)

        with open(merges_filepath) as f:
            merges = deserialize_merge(f)

        Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
