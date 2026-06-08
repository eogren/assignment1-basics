use std::{collections::HashMap, num::NonZeroU32};

use itertools::Itertools;
use rustc_hash::FxHashMap;

pub(crate) trait StatsCollector {
    /// Update the duplicate count for the given sequence index to `dup_count`.
    fn update_duplicate_count(&mut self, sequence_idx: usize, dup_count: u32);

    /// Adjust sequence_idx's count of the given pair by delta. If the stats collector
    /// hasn't tracked any thing for this pair yet, we assume it has a count of 0.
    fn update_pair_count(&mut self, sequence_idx: usize, pair: (u32, u32), delta: i32);
}

#[derive(Debug, Default)]
pub(crate) struct RealStatsCollector {
    /// number of duplicates for each sequence. (eg if hello is inserted
    /// twice, we may dedupe everything and set duplicates[0] to 2)
    duplicates: FxHashMap<u32, u32>,

    /// cache of count of pairs. maps (token, token) -> map of {sequence_idx -> count}
    count_index: FxHashMap<(u32, u32), FxHashMap<usize, u32>>,
}

impl StatsCollector for RealStatsCollector {
    fn update_duplicate_count(&mut self, sequence_idx: usize, dup_count: u32) {
        let seq_idx = u32::try_from(sequence_idx).expect("sequence_idx should fit in u32");

        self.duplicates
            .entry(seq_idx)
            .and_modify(|v| *v = dup_count)
            .or_insert(dup_count);
    }

    fn update_pair_count(&mut self, sequence_idx: usize, pair: (u32, u32), delta: i32) {
        let counts = self.count_index.get_mut(&pair);

        if counts.is_none() {
            if delta < 0 {
                panic!("expect pair to be in index if we are subtracting counts");
            }

            let mut new_map = FxHashMap::default();
            new_map.insert(
                sequence_idx,
                u32::try_from(delta).expect("delta should convert here"),
            );

            self.count_index.insert(pair, new_map);

            return;
        }

        let counts_unwrapped = counts.unwrap();

        counts_unwrapped
            .entry(sequence_idx)
            .and_modify(|e| {
                let new_val = e
                    .checked_add_signed(delta)
                    .expect("never expect underflow here");
                *e = new_val;
            })
            .or_insert_with(|| u32::try_from(delta).expect("delta should convert here"));
    }
}

impl RealStatsCollector {
    fn sequences_with_pair(&self, pair: (u32, u32)) -> Vec<usize> {
        self.count_index
            .get(&pair)
            .map(|counts| counts.keys().copied().collect_vec())
            .unwrap_or_default()
    }

    pub fn counts(&self) -> Vec<CountInfo> {
        let mut ret = Vec::new();

        for (k, v) in self.count_index.iter() {
            let dup_count: u32 = v.iter().fold(0, |acc, e| {
                let sequence_index = u32::try_from(*e.0).expect("sequence_idx should fit in u32");
                let num_occurrences = *e.1;

                acc + (num_occurrences
                    * self
                        .duplicates
                        .get(&sequence_index)
                        .expect("sequence_index should exist in duplicates list"))
            });
            if dup_count > 0 {
                ret.push(CountInfo {
                    token_pair: *k,
                    count: dup_count,
                });
            }
        }

        ret
    }
}

#[derive(Debug, Default)]
pub(crate) struct NoOpStatsCollector {}

impl StatsCollector for NoOpStatsCollector {
    fn update_duplicate_count(&mut self, _sequence_idx: usize, _dup_count: u32) {}

    fn update_pair_count(&mut self, _sequence_idx: usize, _pair: (u32, u32), _delta: i32) {}
}

/// SequenceShard keeps track of a set of token sequences
/// and handles the logic around merging sequences together.
#[derive(Debug, Default)]
pub(crate) struct SequenceShard<S: StatsCollector> {
    /// raw data for all the sequences. this will eventually
    /// be manipulated when merges occur.
    sequences: Vec<u32>,

    /// the next token for sequences[i]. this is simulating
    /// a double linked list and will be used for merges.
    next_token: Vec<Option<NonZeroU32>>,

    /// starting index for each sequence in the shard. points
    /// at sequences[i].
    start_index: Vec<u32>,

    /// stats collector
    stats_collector: S,
}

/// Iterate through pairs in a single sequence
#[derive(Debug)]
pub(crate) struct SequenceCursor<'a, S: StatsCollector> {
    shard: &'a mut SequenceShard<S>,

    /// which sequence in the shard is this
    sequence_idx: usize,

    /// Sequence index before the pair we are pointing at. Starts out as None
    idx_before_pair: Option<usize>,
}

impl<'a, S: StatsCollector> SequenceCursor<'a, S> {
    pub fn new(shard: &mut SequenceShard<S>, sequence_idx: usize) -> SequenceCursor<'_, S> {
        debug_assert!(
            shard.sequences.len() > sequence_idx,
            "sequence_idx not present in shard"
        );

        SequenceCursor {
            shard,
            sequence_idx,
            idx_before_pair: None,
        }
    }

    /// Return the pair this cursor is pointing at.
    /// Returns None if there are no pairs left to visit.
    pub fn current_pair(&self) -> Option<(u32, u32)> {
        let first_pair_char_idx = self.get_sequence_index_of_current_pair()?;

        let second_pair_char_idx = self.shard.next_token[first_pair_char_idx]
            .map(|s| usize::try_from(s.get()).expect("u32 should always convert to usize"))?;

        Some((
            self.shard.sequences[first_pair_char_idx],
            self.shard.sequences[second_pair_char_idx],
        ))
    }

    /// Advance the cursor to the next pair element (or do nothin if it's already
    /// at the end). If the cursor is now at the end will also return false.
    pub fn next(&mut self) -> bool {
        if let Some(idx) = self.get_sequence_index_of_current_pair() {
            self.idx_before_pair = Some(idx)
        }

        self.is_done()
    }

    /// Is this cursor at the end?
    pub fn is_done(&self) -> bool {
        self.current_pair().is_none()
    }

    /// Merge the current pair into a new token and update the cursor accordingly.
    /// Eg, if our sequence is (1, 2, 3, 4) and the cursor is pointing at the (1, 2) pair -
    /// merge(5) will update the sequence to (5, 3, 4). The cursor will point at (5, 3).
    pub fn merge_pair(&mut self, new_token: u32) {
        let first_pair_idx = self
            .get_sequence_index_of_current_pair()
            .expect("first_pair_idx should always exist");

        let second_pair_idx = self.shard.next_token[first_pair_idx]
            .map(|s| s.get() as usize)
            .expect("second_pair_idx should always exist");

        let next_char_idx = self.shard.next_token[second_pair_idx];

        // Get information about characters to merge before we change anything
        let before_pair_char = self.idx_before_pair.map(|i| self.shard.sequences[i]);
        let first_pair_char = self.shard.sequences[first_pair_idx];
        let second_pair_char = self.shard.sequences[second_pair_idx];
        let after_pair_char = next_char_idx.map(|i| self.shard.sequences[i.get() as usize]);

        // Replace character and update next pointer to fourth charcater in sequence (eg, '4' in
        // example above)
        self.shard.sequences[first_pair_idx] = new_token;
        self.shard.next_token[first_pair_idx] = next_char_idx;

        // Now update counts
        // 1. Decrement count of ([before-pair], [first-element-of-pair] if before-pair exists)
        // and increment count of [before-pair], [new-token]
        if let Some(before_char) = before_pair_char {
            self.update_count((before_char, first_pair_char), -1);
            self.update_count((before_char, new_token), 1);
        }

        // 2. Decrement count of [first-element, second-element]
        self.update_count((first_pair_char, second_pair_char), -1);

        // 3. Decrement count of [second-element], [after-pair]
        //    and increment [new-token], [after-pair]
        if let Some(after_char) = after_pair_char {
            self.update_count((second_pair_char, after_char), -1);
            self.update_count((new_token, after_char), 1);
        }
    }

    fn update_count(&mut self, pair: (u32, u32), delta: i32) {
        self.shard
            .stats_collector
            .update_pair_count(self.sequence_idx, pair, delta);
    }

    fn get_sequence_index_of_current_pair(&self) -> Option<usize> {
        match self.idx_before_pair {
            Some(idx) => self.shard.next_token[idx]
                .map(|s| usize::try_from(s.get()).expect("u32 should always convert to usize")),

            None => self
                .shard
                .start_index
                .get(self.sequence_idx)
                .map(|s| *s as usize),
        }
    }

    fn into_pairs(mut self) -> Vec<(u32, u32)> {
        let mut ret = Vec::new();

        while !self.is_done() {
            ret.push(self.current_pair().expect("pair should be valid"));
            self.next();
        }

        ret
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct CountInfo {
    pub token_pair: (u32, u32),
    pub count: u32,
}

impl CountInfo {
    pub fn compare_to(
        &self,
        other: &CountInfo,
        token_dict: &HashMap<u32, Vec<u8>>,
    ) -> std::cmp::Ordering {
        if self.count > other.count {
            return std::cmp::Ordering::Greater;
        }

        if self.count < other.count {
            return std::cmp::Ordering::Less;
        }

        self.get_token_tuple(token_dict)
            .cmp(&other.get_token_tuple(token_dict))
    }

    fn get_token_tuple<'a>(&self, token_dict: &'a HashMap<u32, Vec<u8>>) -> (&'a [u8], &'a [u8]) {
        (
            token_dict
                .get(&self.token_pair.0)
                .expect("first token in pair should exist"),
            token_dict
                .get(&self.token_pair.1)
                .expect("second pair in token should exist"),
        )
    }
}

impl<S: StatsCollector> SequenceShard<S> {
    pub fn new(stats: S) -> Self {
        Self {
            sequences: Vec::new(),
            next_token: Vec::new(),
            start_index: Vec::new(),
            stats_collector: stats,
        }
    }

    /// Add a given sequence to this shard. Sequence is a series of tokens.
    pub fn push(&mut self, sequence: &[u32], dup_count: u32) {
        assert!(dup_count > 0, "dup_count cannot be zero");
        assert!(!sequence.is_empty(), "sequence cannot be empty");

        // start_idx = index into self.sequences<> where the
        // sequence will start.
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
        let seq_idx = self.start_index.len();

        self.start_index.push(start_idx);
        self.stats_collector
            .update_duplicate_count(seq_idx, dup_count);

        // 3: Every token has next of i + 1 except for last in the sequence
        // which should have last of None
        for i in start_idx..start_idx + sequence_len - 1 {
            self.next_token.push(NonZeroU32::new(i + 1));
        }

        self.next_token.push(None);
        self.update_counts(
            u32::try_from(self.start_index.len() - 1).expect("expect index to fit in u32"),
        );
    }

    /// Return a cursor over a specific sequence in this shard
    pub fn cursor_mut(&mut self, sequence_idx: usize) -> SequenceCursor<'_, S> {
        SequenceCursor::new(self, sequence_idx)
    }

    /// Convert this shard into a sequence of every token it contains.
    pub fn into_tokens(mut self) -> Vec<u32> {
        let mut used = 0;

        // walk through everything in the sequence array, collapsing
        // next pointer
        let num_sequences = self.start_index.len();
        for sequence in 0..num_sequences {
            let mut char_cursor_idx = self.start_index[sequence];
            loop {
                debug_assert!(
                    used <= char_cursor_idx as usize,
                    "should always be collapsing values"
                );
                self.sequences[used] = self.sequences[char_cursor_idx as usize];
                used += 1;

                match self.next_token[char_cursor_idx as usize] {
                    None => break,
                    Some(next_idx) => char_cursor_idx = next_idx.get(),
                }
            }
        }

        // then condense vector
        self.sequences.resize(used, 0);
        self.sequences
    }

    /// Update counts for the sequence_idx that was just recently added to this shard
    fn update_counts(&mut self, sequence_idx: u32) {
        let c = self.cursor_mut(usize::try_from(sequence_idx).expect("sequence should fit in u32"));
        let pairs = c.into_pairs();
        for pair in pairs.into_iter() {
            self.stats_collector
                .update_pair_count(sequence_idx as usize, pair, 1);
        }
    }
}

impl SequenceShard<RealStatsCollector> {
    pub fn counts(&self) -> Vec<CountInfo> {
        self.stats_collector.counts()
    }

    #[tracing::instrument(skip(self), fields(pair, new_token))]
    /// Merge all instances of `pair` together, replacing them with new_token
    pub fn merge_pair(&mut self, pair: (u32, u32), new_token: u32) {
        let sequences_with_pair = self.stats_collector.sequences_with_pair(pair);
        for sequence_index in sequences_with_pair {
            let mut c = self.cursor_mut(sequence_index);
            while !c.is_done() {
                let c_pair = c.current_pair().expect("current_pair should be valid");
                if c_pair == pair {
                    c.merge_pair(new_token);
                } else {
                    c.next();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering::{Greater, Less};

    use super::*;

    #[test]
    pub fn test_basic_add() {
        let mut s = SequenceShard::new(RealStatsCollector::default());

        s.push(&[1, 1, 1, 1], 2);
        let counts = s.counts();

        assert_eq!(counts.len(), 1);
        assert_eq!(
            *counts.first().unwrap(),
            CountInfo {
                token_pair: (1, 1),
                count: 6
            }
        );
    }

    /// Dump all pairs in a SequenceCursor
    fn to_pairs<S: StatsCollector>(c: &mut SequenceCursor<S>) -> Vec<(u32, u32)> {
        let mut ret = Vec::new();

        while !c.is_done() {
            ret.push(c.current_pair().expect("expect a valid pair when not done"));
            c.next();
        }

        ret
    }

    #[test]
    pub fn test_cursor_pair() {
        let mut s = SequenceShard::new(RealStatsCollector::default());

        s.push(&[1, 1, 2, 3], 2);

        let mut c = s.cursor_mut(0);
        assert_eq!(to_pairs(&mut c), vec![(1, 1), (1, 2), (2, 3)]);
    }

    #[test]
    pub fn test_two_sequences() {
        let mut s = SequenceShard::new(RealStatsCollector::default());

        s.push(&[3, 1, 1, 2], 2);
        s.push(&[4, 1, 1, 5], 1);

        {
            let counts = s.counts();
            // [3, 1] = 2
            // [1, 1] = 2+1 = 3
            // [1, 2] = 2
            // [4, 1] = 1
            // [1, 5] = 1

            assert_eq!(counts.len(), 5);
            let one_one = counts
                .iter()
                .find(|ci| ci.token_pair == (1, 1))
                .expect("expected to find 1,1");
            assert_eq!(one_one.count, 3);
        }

        s.merge_pair((1, 1), 6);
        {
            let counts = s.counts();
            // [3, 6] = 2
            // [6, 2] = 2
            // [4, 6] = 1
            // [6, 5] = 1

            assert_eq!(counts.len(), 4);
            let three_six = counts
                .iter()
                .find(|ci| ci.token_pair == (3, 6))
                .expect("expected to find 3, 6");
            assert_eq!(three_six.count, 2);
        }
    }

    #[test]
    pub fn test_merge_simpler() {
        let mut s = SequenceShard::new(RealStatsCollector::default());

        s.push(&[3, 1, 1, 2], 2);
        s.merge_pair((1, 1), 4);

        let tokens = s.into_tokens();
        assert_eq!(tokens, vec![3, 4, 2]);
    }

    #[test]
    pub fn test_merge_runs() {
        let mut s = SequenceShard::new(RealStatsCollector::default());

        s.push(&[1, 1, 1, 1, 1], 2);

        // Greedy merge:
        // [1, 1, 1, 1, 1] -> [2, 1, 1, 1] -> [2, 2, 1]
        s.merge_pair((1, 1), 2);
        let counts = s.counts();

        let mut pairs_one_two_found = false;
        let mut pairs_two_founds = false;
        assert_eq!(counts.len(), 2);

        for count in counts {
            if count.token_pair == (2, 2) {
                assert_eq!(count.count, 2);
                pairs_two_founds = true;
            } else if count.token_pair == (2, 1) {
                assert_eq!(count.count, 2);
                pairs_one_two_found = true;
            } else {
                assert!(
                    false,
                    "Unexpected token pair ({}, {})",
                    count.token_pair.0, count.token_pair.1
                );
            }
        }

        assert!(pairs_one_two_found, "Didn't find (2,1) pair");
        assert!(pairs_two_founds, "Didn't find (2,2) pair");
    }

    #[test]
    fn test_compare_by_count() {
        let token_dict = HashMap::new();
        let c1 = CountInfo {
            token_pair: (1, 2),
            count: 5,
        };

        let c2 = CountInfo {
            token_pair: (3, 4),
            count: 4,
        };

        assert_eq!(c1.compare_to(&c2, &token_dict), Greater);
        assert_eq!(c2.compare_to(&c1, &token_dict), Less);
    }

    #[test]
    fn test_compare_lexographically() {
        let mut token_dict = HashMap::new();
        token_dict.insert(1, b"ab".to_vec());
        token_dict.insert(2, b"c".to_vec());
        token_dict.insert(3, b"a".to_vec());
        token_dict.insert(4, b"bc".to_vec());

        let c1 = CountInfo {
            token_pair: (1, 2),
            count: 5,
        };

        let c2 = CountInfo {
            token_pair: (3, 4),
            count: 5,
        };

        // Concrete divergence:
        //  - Pair A = (b"ab", b"c") → concat b"abc"
        //  - Pair B = (b"a", b"bc") → concat b"abc"
        //
        //  Concatenated, they're equal. As tuples, A > B (because b"ab" > b"a"). The Python reference compares
        //  tuples — (b'e', b'l') vs (b'l', b'e') — so for spec-correctness you want tuple ordering, not
        //  concatenated.
        assert_eq!(c1.compare_to(&c2, &token_dict), Greater);
        assert_eq!(c2.compare_to(&c1, &token_dict), Less);
    }
}
