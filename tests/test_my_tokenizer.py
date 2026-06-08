import cs336_basics.tokenizer
import pytest

from io import StringIO

from cs336_basics.tokenizer import deserialize_merge, deserialize_vocab, serialize_merges, serialize_vocab


def test_is_printable():
    from cs336_basics.tokenizer import _is_printable

    assert not _is_printable(0)
    assert _is_printable(0x21)
    assert _is_printable(0x7E)
    assert _is_printable(0xA1)
    assert _is_printable(0xAC)
    assert _is_printable(0xAE)
    assert _is_printable(0xFF)


def test_vocab_serde():
    counter = 0
    vocab = {}

    # Generate every 2 byte combo we have as part of vocab
    for i in range(0, 256):
        vocab[counter] = bytes([i])
        counter = counter + 1

    for i in range(0, 256):
        for j in range(0, 256):
            vocab[counter] = bytes([i, j])
            counter = counter + i

    output_buf = StringIO()
    serialize_vocab(output_buf, vocab)
    output_buf.seek(0)
    rtt_vocab = deserialize_vocab(output_buf)

    assert vocab == rtt_vocab


def test_merge_serde():
    orig_merges = []

    # Generate every 2 byte combo we have as part of vocab
    for first_i in range(0, 256):
        first = bytes(first_i)
        for second_i in range(0, 10):
            for second_j in range(25, 35):
                second = bytes([second_i, second_j])
                orig_merges.append((first, second))

    output_buf = StringIO()
    serialize_merges(output_buf, orig_merges)
    output_buf.seek(0)
    rtt_merges = deserialize_merge(output_buf)

    assert orig_merges == rtt_merges


def test_validate_vocab():
    good_vocab = {1: b"foo"}
    bad_vocab = {1: b"foo", 2: b"foo"}

    cs336_basics.tokenizer._validate_vocab(good_vocab)
    with pytest.raises(ValueError):
        cs336_basics.tokenizer._validate_vocab(bad_vocab)


def test_convert_merges():
    vocab = {1: b"a", 2: b"b", 3: b"c", 4: "d", 5: b"ab", 6: b"abc"}
    reverse_vocab = {v: k for k, v in vocab.items()}
    merges = [(b"a", b"b"), (b"ab", b"c")]

    token_merges = cs336_basics.tokenizer._convert_merges_to_tokens(reverse_vocab, merges)
    assert token_merges == [(1, 2, 5), (5, 3, 6)]
