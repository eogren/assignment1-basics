/// Keeps track of the sequences discovered during pretokenization in a memory friendly way
use std::sync::OnceLock;

use fancy_regex::Regex;
use rustc_hash::FxHashMap;

static GPT2_REGEX: OnceLock<Regex> = OnceLock::new();

fn gpt_regex() -> &'static Regex {
    GPT2_REGEX.get_or_init(|| {
        Regex::new(r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)
            .expect("expected gpt2 regex to compile")
    })
}

fn split_regex(special_tokens: &[&str]) -> Regex {
    let re = itertools::join(
        special_tokens
            .iter()
            .map(|token| fancy_regex::escape(token)),
        "|",
    );
    Regex::new(&re).expect("expected split_regex to compile")
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
/*
pub(crate) struct Sequences {
    /// Raw data of the sequence as a series of bytes (or tokens)
    data: Vec<u8>,

    /// Offsets into the data array that define each sequence
    offsets: Vec<usize>,

    /// Number of times said sequence appears in the text
    counts: Vec<usize>,
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct SequenceInfo<'a> {
    pub sequence: &'a [u8],
    pub count: usize,
}

impl Sequences {
    pub fn new() -> Self {
        Sequences {
            data: Vec::new(),
            offsets: Vec::new(),
            counts: Vec::new(),
        }
    }

    pub fn with_estimated_data(size: usize) -> Self {
        let mut s = Self::new();
        s.data.reserve(size);

        s
    }

    pub fn iter(&self) -> SequencesIter<'_> {
        SequencesIter {
            sequence: self,
            pos_idx: 0,
        }
    }

    /// Append bytes into the sequences structure
    pub fn append(&mut self, bytes: &[u8]) {
        // TODO: Dedup
        self.data.extend_from_slice(bytes);
        self.offsets.push(self.data.len());
        self.counts.push(1);
    }
}

pub(crate) struct SequencesIter<'a> {
    sequence: &'a Sequences,
    pos_idx: usize,
}

impl<'a> Iterator for SequencesIter<'a> {
    type Item = SequenceInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.sequence.offsets.get(self.pos_idx) {
            Some(idx) => {
                let start_idx = match self.pos_idx {
                    0 => 0,
                    _ => self.sequence.offsets[self.pos_idx - 1],
                };

                let r = Some(SequenceInfo {
                    sequence: &self.sequence.data[start_idx..self.sequence.offsets[self.pos_idx]],
                    count: self.sequence.counts[self.pos_idx],
                });
                self.pos_idx = self.pos_idx + 1;

                r
            }
            None => None,
        }
    }
}
*/

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

    /*
    #[test]
    pub fn test_iterate_through_empty_sequence() {
        let s = Sequences::new();
        let iter = s.iter();

        assert_eq!(0, iter.count());
    }

    #[test]
    pub fn test_insert_different_values() {
        let mut s = Sequences::new();
        s.append(b"Hello");
        s.append(b"World");

        let iter = s.iter();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                SequenceInfo {
                    count: 1,
                    sequence: b"Hello"
                },
                SequenceInfo {
                    count: 1,
                    sequence: b"World"
                }
            ]
        );
    }

    #[test]
    pub fn test_with_dupes() {
        let mut s = Sequences::new();
        s.append(b"Hello");
        s.append(b"World");
        s.append(b"Hello");

        let iter = s.iter();
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                SequenceInfo {
                    count: 2,
                    sequence: b"Hello"
                },
                SequenceInfo {
                    count: 1,
                    sequence: b"World"
                }
            ]
        );
    }

    */
}
