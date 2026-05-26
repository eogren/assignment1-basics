use std::num::NonZeroU32;

use rustc_hash::FxHashMap;

/// SequenceShard keeps track of a set of token sequences
/// and handles the logic around merging sequences together.
pub(crate) struct SequenceShard {
    /// raw data for all the sequences. this will eventually
    /// be manipulated when merges occur.
    sequences: Vec<u32>,

    /// the next token for sequences[i]. this is simulating
    /// a double linked list and will be used for merges.
    next_token: Vec<Option<NonZeroU32>>,

    /// starting index for each sequence in the shard. points
    /// at sequences[i].
    start_index: Vec<u32>,

    /// cache of count of pairs. trading off some memory for
    /// faster compute. maps (token, token) -> set{token indices where that pair starts, index into the sequence}
    count_index: FxHashMap<(u32, u32), Vec<(u32, u32)>>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct CountInfo {
    pub token_pair: (u32, u32),
    pub count: u32,
}

impl SequenceShard {
    pub fn new() -> Self {
        return Self::default();
    }

    /// Add a given sequence to this shard. Sequence is a series of tokens.
    pub fn push(&mut self, sequence: &[u32]) {
        let start_idx = self
            .sequences
            .len()
            .try_into()
            .expect("can only store u32 worth of indices");
        let sequence_len: u32 = sequence
            .len()
            .try_into()
            .expect("can only store u32 of one sequence");

        // 1: Append actual sequence
        self.sequences.extend_from_slice(sequence);

        // 2: Start index point to beginning of sequence we just pushed
        self.start_index.push(start_idx);

        // 3: Every token has next of i + 1 except for last in the sequence
        // which should have last of None
        for i in start_idx..start_idx + sequence_len - 1 {
            self.next_token.push(NonZeroU32::new(i + 1));
        }

        self.next_token.push(NonZeroU32::new(0));
        self.update_counts(start_idx);
    }

    /// Retrieve the counts of each pair of tokens in this shard.
    /// E.g. if we have 'l', 'o', 'w' [pretend these are token ids],
    /// should return ('l', 'o) -> 1, ('o', 'w') -> 1
    pub fn counts(&self) -> Vec<CountInfo> {
        let mut ret = Vec::new();

        for (k, v) in self.count_index.iter() {
            ret.push(CountInfo {
                token_pair: *k,
                count: v.len().try_into().expect("only u32 counts"),
            });
        }

        ret
    }

    fn update_counts(&mut self, sequence_idx: u32) {
        let mut start_idx = 0;
        loop {
            let pair = self.get_pair(sequence_idx, start_idx);
            if pair.is_none() {
                break;
            }

            self.count_index
                .entry(pair.unwrap())
                .and_modify(|e| e.push((sequence_idx, start_idx)))
                .or_insert_with(|| vec![(sequence_idx, start_idx)]);

            start_idx += 1;
        }
    }

    /// Return the pair for sequence {sequence_idx} at {first_token_idx}. If {first_token_idx}
    /// is the last token in the sequence, return None instead.
    ///
    /// TODO: this works but is inefficient, we keep traversing the linked list for each character
    /// in the sequence. might be fine since it's probably still cached and sequences are short.
    fn get_pair(&self, sequence_idx: u32, first_token_idx: u32) -> Option<(u32, u32)> {
        let mut idx = self.start_index[sequence_idx as usize];
        for _i in 0..first_token_idx {
            idx = self.next_token[idx as usize]
                .expect("first_token_idx should always be valid")
                .into();
        }

        let first_token = self.sequences[idx as usize];
        match self.next_token[idx as usize] {
            Some(second_idx) => Some((first_token, self.sequences[second_idx.get() as usize])),
            None => None,
        }
    }
}

impl Default for SequenceShard {
    fn default() -> Self {
        Self {
            sequences: Vec::new(),
            start_index: Vec::new(),
            next_token: Vec::new(),
            count_index: FxHashMap::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_basic_add() {
        let mut s = SequenceShard::new();

        s.push(&[1, 1, 1, 1]);
        let counts = s.counts();

        assert_eq!(counts.len(), 1);
        assert_eq!(
            *counts.first().unwrap(),
            CountInfo {
                token_pair: (1, 1),
                count: 3
            }
        );
    }
}
