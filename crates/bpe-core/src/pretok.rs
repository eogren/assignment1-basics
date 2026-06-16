/// Keeps track of the sequences discovered during pretokenization in a memory friendly way
use std::sync::OnceLock;

use fancy_regex::Regex;
use itertools::Itertools;
use rustc_hash::FxHashMap;

use crate::sequence::SequenceShard;

static GPT2_REGEX: OnceLock<Regex> = OnceLock::new();

fn gpt_regex() -> &'static Regex {
    GPT2_REGEX.get_or_init(|| {
        Regex::new(r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)
            .expect("expected gpt2 regex to compile")
    })
}

fn split_regex(special_tokens: &[&str]) -> Regex {
    // Put the longest tokens first so rust's engine will match them. (eg if aa and a are both special tokens,
    // we want to match aa when possible)
    let mut indices: Vec<usize> = (0..special_tokens.len()).collect();
    indices.sort_by(|&a, &b| special_tokens[b].cmp(special_tokens[a]));

    let re = itertools::join(
        indices
            .into_iter()
            .map(|i| fancy_regex::escape(special_tokens[i])),
        "|",
    );
    Regex::new(&re).expect("expected split_regex to compile")
}

pub(crate) fn pretokenize_chunk_for_encoding(
    chunk: &[u8],
    special_tokens: &[(&str, u32)],
    // reverse_vocab is a slice [0-255] that indexes a byte into its token.
    reverse_vocab: &[Option<u32>],
) -> SequenceShard {
    let token_strs = special_tokens.iter().map(|i| i.0).collect_vec();
    let split_re = split_regex(&token_strs);
    let chunk_str = std::str::from_utf8(chunk).expect("expect always utf8");

    let splits = split_re.find_iter(chunk_str);

    let mut last_idx = 0;
    let mut ret = SequenceShard::new();
    for split in splits.into_iter() {
        let tok_match = split.expect("no error expected in regex");
        let token_str = &chunk_str[tok_match.range()];
        if token_str.is_empty() {
            continue;
        }
        let token_idx = special_tokens
            .iter()
            .find(|item| item.0 == token_str)
            .expect("expected to find match for token")
            .1;

        // Pretok everything before this and add to shard
        let pre_tok_str = &chunk_str[last_idx..tok_match.start()];
        preokenize_slice_for_encoding(&mut ret, pre_tok_str, reverse_vocab);

        ret.push(&[token_idx], 1);
        last_idx = tok_match.end();
    }

    if last_idx != chunk_str.len() {
        let last_slice = &chunk_str[last_idx..chunk_str.len()];
        preokenize_slice_for_encoding(&mut ret, last_slice, reverse_vocab);
    }
    ret
}

fn preokenize_slice_for_encoding(
    ret: &mut SequenceShard,
    pre_tok_str: &str,
    reverse_vocab: &[Option<u32>],
) {
    let gpt_reg = gpt_regex();
    let tokens = gpt_reg.find_iter(pre_tok_str);
    for token in tokens {
        let tokenized = token
            .expect("expected no error")
            .as_str()
            .as_bytes()
            .iter()
            .map(|b| reverse_vocab[*b as usize].expect("should have a token for char"))
            .collect_vec();
        ret.push(&tokenized, 1);
    }
}

/// Pretokenize a chunk of text and return its associated SequenceBuilder.
pub(crate) fn pretokenize_chunk_for_training(
    chunk: &[u8],
    special_tokens: &[&str],
) -> SequenceBuilder {
    // Step 1: Split and remove special tokens
    let split_re = split_regex(special_tokens);
    let chunk_str = std::str::from_utf8(chunk).expect("expect always utf8");

    let splits = split_re.split(chunk_str);
    let gpt_reg = gpt_regex();

    let mut ret = SequenceBuilder::new();

    for split in splits {
        let tokens = gpt_reg.find_iter(split.expect("expect split to always succeed"));
        for token in tokens {
            ret.append(token.expect("expect token to be valid").as_str().as_bytes());
        }
    }

    ret
}

#[derive(Debug, PartialEq)]
pub(crate) struct SequenceBuilder {
    sequences: FxHashMap<Vec<u8>, usize>,
}

impl SequenceBuilder {
    pub fn new() -> Self {
        SequenceBuilder {
            sequences: FxHashMap::default(),
        }
    }

    /// Merge two sequence builders together, consuming them both and returning a merged copy.
    /// The merged copy shoul dhave all the sequences in both; with counts added together
    /// where there was overlap.
    pub fn merge(s1: SequenceBuilder, s2: SequenceBuilder) -> SequenceBuilder {
        let (smallest, mut biggest) = if s1.counts().len() < s2.counts().iter().len() {
            (s1, s2)
        } else {
            (s2, s1)
        };

        let smallest_items = smallest.into_counts();

        for item in smallest_items {
            biggest
                .sequences
                .entry(item.0)
                .and_modify(|e| *e += item.1)
                .or_insert(item.1);
        }
        biggest
    }

    pub fn append(&mut self, bytes: &[u8]) {
        match self.sequences.get_mut(bytes) {
            None => {
                self.sequences.insert(bytes.to_vec(), 1);
            }
            Some(count) => {
                *count += 1;
            }
        }
    }

    pub fn counts(&self) -> &FxHashMap<Vec<u8>, usize> {
        &self.sequences
    }

    fn into_counts(self) -> FxHashMap<Vec<u8>, usize> {
        self.sequences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_sequence_builders() -> (SequenceBuilder, SequenceBuilder) {
        let mut s = SequenceBuilder::new();

        s.append(b"Hello");
        s.append(b"world");
        s.append(b"Hello");

        let mut s2 = SequenceBuilder::new();

        s2.append(b"Hi");
        s2.append(b"world");

        (s, s2)
    }

    #[test]
    pub fn test_sequence_builder() {
        let mut s = SequenceBuilder::new();

        s.append(b"Hello");
        s.append(b"world");
        s.append(b"Hello");

        let c = s.counts();
        assert_eq!(c.len(), 2);

        assert_eq!(c.get(b"Hello".as_slice()).copied(), Some(2));
        assert_eq!(c.get(b"world".as_slice()).copied(), Some(1));
    }

    #[test]
    pub fn test_sequence_builder_merge() {
        let (s1, s2) = get_sequence_builders();

        let merged = SequenceBuilder::merge(s1, s2);
        assert_eq!(merged.counts().len(), 3);
        assert_eq!(
            *merged
                .counts()
                .get(b"Hello".as_slice())
                .expect("expected Hello in map"),
            2
        );
        assert_eq!(
            *merged
                .counts()
                .get(b"world".as_slice())
                .expect("expected world in map"),
            2
        );
        assert_eq!(
            *merged
                .counts()
                .get(b"Hi".as_slice())
                .expect("expected hi in map"),
            1
        );
    }

    #[test]
    pub fn test_merge_symmetric() {
        let (first_s1, first_s2) = get_sequence_builders();
        let first_merged = SequenceBuilder::merge(first_s1, first_s2);

        let (second_s1, second_s2) = get_sequence_builders();
        let second_merged = SequenceBuilder::merge(second_s2, second_s1);

        assert_eq!(first_merged, second_merged);
    }

    #[test]
    pub fn test_regex_compiles() {
        gpt_regex();
    }

    #[test]
    pub fn test_split_regex_escapes() {
        let splits = vec!["<|endoftext|>", "<|otherthing|>"];
        let s = "Hello world!<|endoftext|> <|otherthing|> Here are my tokens";

        let binding = split_regex(&splits);
        let splits = binding.split(s).map(|x| x.unwrap());
        let expected = vec!["Hello world!", " ", " Here are my tokens"];

        itertools::assert_equal(splits, expected);
    }

    #[test]
    pub fn test_tokenize_chunk() {
        let special_tokens = vec!["<|endoftext|>", "<|otherthing|>"];
        let s = "Hello world!<|endoftext|> <|otherthing|> Here are my tokens";

        let builder = pretokenize_chunk_for_training(s.as_bytes(), &special_tokens);
        let counts = builder.counts();
        assert_eq!(counts.len(), 8);
        assert!(counts.contains_key(b"!".as_slice()));
        assert!(counts.contains_key(b" tokens".as_slice()));
    }

    #[test]
    pub fn test_apost() {
        let special_tokens = vec!["<|endoftext|>"];
        let s = "don't students think";

        let builder = pretokenize_chunk_for_training(s.as_bytes(), &special_tokens);
        let counts = builder.counts();
        assert_eq!(counts.len(), 4);
        assert!(counts.contains_key(b"don".as_slice()));
        assert!(counts.contains_key(b"'t".as_slice()));
        assert!(counts.contains_key(b" students".as_slice()));
        assert!(counts.contains_key(b" think".as_slice()));
    }

    static DEFAULT_REVERSE_VOCAB: OnceLock<[Option<u32>; 256]> = OnceLock::new();
    fn reverse_vocab() -> &'static [Option<u32>] {
        return DEFAULT_REVERSE_VOCAB.get_or_init(|| {
            let ret: [Option<u32>; 256] =
                std::array::from_fn(|i| Some(u32::try_from(i).expect("u32 should fit")));

            ret
        });
    }

    #[test]
    pub fn test_encoding_tokenizer_no_end_token() {
        let special_tokens = vec![("<|endoftext|>", 257)];
        let s = "don't students";
        let expected_tokens = s.as_bytes().iter().map(|c| *c as u32).collect_vec();

        let builder =
            pretokenize_chunk_for_encoding(s.as_bytes(), &special_tokens, reverse_vocab());
        let tokens = builder.into_tokens();
        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    pub fn test_encoding_tokenizer_with_double_token() {
        let special_tokens = vec![("<|endoftext|>", 257), ("<|endoftext|><|endoftext|>", 258)];
        let s = "don't students<|endoftext|><|endoftext|>";
        let mut expected_tokens = b"don't".iter().map(|c| *c as u32).collect_vec();
        b" students"
            .iter()
            .for_each(|c| expected_tokens.push(*c as u32));

        let builder =
            pretokenize_chunk_for_encoding(s.as_bytes(), &special_tokens, reverse_vocab());
        let tokens = builder.into_tokens();
        expected_tokens.push(258);
        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    pub fn test_encoding_tokenizer_with_end_token() {
        let special_tokens = vec![("<|endoftext|>", 257)];
        let s = "don't<|endoftext|>students";
        let mut expected_tokens = b"don't".iter().map(|c| *c as u32).collect_vec();
        expected_tokens.push(257);
        b"students"
            .iter()
            .for_each(|c| expected_tokens.push(*c as u32));

        let builder =
            pretokenize_chunk_for_encoding(s.as_bytes(), &special_tokens, reverse_vocab());
        let tokens = builder.into_tokens();
        assert_eq!(tokens, expected_tokens);
    }
}
