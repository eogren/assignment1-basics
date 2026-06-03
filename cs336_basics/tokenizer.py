from __future__ import annotations

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


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        pass

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
