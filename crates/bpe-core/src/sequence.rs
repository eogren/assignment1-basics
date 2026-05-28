use std::num::NonZeroU32;

use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct CountVal {
    pub sequence_index: u32,
    pub num_occurrences: u32,
}

/// SequenceShard keeps track of a set of token sequences
/// and handles the logic around merging sequences together.
#[derive(Debug)]
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

    /// number of duplicates for each sequence. (eg if hello is inserted
    /// twice, we may dedupe everything and set duplicates[0] to 2)
    duplicates: Vec<u32>,

    /// cache of count of pairs. trading off some memory for
    /// faster compute. maps (token, token) -> set{sequence_idx, count}
    count_index: FxHashMap<(u32, u32), Vec<CountVal>>,
}

/// Iterate through pairs in a single sequence
#[derive(Debug)]
pub(crate) struct SequenceCursor<'a> {
    shard: &'a mut SequenceShard,

    /// which sequence in the shard is this
    sequence_idx: usize,

    /// Sequence index before the pair we are pointing at. Starts out as None
    idx_before_pair: Option<usize>,
}

impl<'a> SequenceCursor<'a> {
    pub fn new(shard: &mut SequenceShard, sequence_idx: usize) -> SequenceCursor<'_> {
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
        match self.get_sequence_index_of_current_pair() {
            Some(idx) => self.idx_before_pair = Some(idx),
            None => (),
        }
        self.is_done()
    }

    /// Is this cursor at the end?
    pub fn is_done(&self) -> bool {
        self.current_pair() == None
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

    /// Update the count index for given pair.
    fn update_count(&mut self, pair: (u32, u32), delta: i32) {
        let counts = self.shard.count_index.get_mut(&pair);

        if counts.is_none() {
            if delta < 0 {
                assert!(
                    false,
                    "expect pair to be in index if we are subtracting counts"
                );
            }

            self.shard.count_index.insert(
                pair,
                vec![CountVal {
                    sequence_index: u32::try_from(self.sequence_idx)
                        .expect("sequence should fit in u32"),
                    num_occurrences: 1,
                }],
            );

            return;
        }

        let my_count = counts
            .unwrap()
            .iter_mut()
            .find(|ci| ci.sequence_index as usize == self.sequence_idx)
            .expect("should always find this sequence in index");

        let new_count = my_count
            .num_occurrences
            .checked_add_signed(delta)
            .expect("should not under/overflow");
        my_count.num_occurrences = new_count;
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

impl SequenceShard {
    pub fn new() -> Self {
        return Self::default();
    }

    /// Add a given sequence to this shard. Sequence is a series of tokens.
    pub fn push(&mut self, sequence: &[u32], dup_count: u32) {
        assert!(dup_count > 0, "dup_count cannot be zero");
        assert!(sequence.len() > 0, "sequence cannot be empty");

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
        self.start_index.push(start_idx);
        self.duplicates.push(dup_count);

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
    pub fn cursor_mut(&mut self, sequence_idx: usize) -> SequenceCursor<'_> {
        SequenceCursor::new(self, sequence_idx)
    }

    #[tracing::instrument]
    /// Merge all instances of `pair` together, replacing them with new_token
    pub fn merge_pair(&mut self, pair: (u32, u32), new_token: u32) {
        let sequences_with_pair = self.count_index.get(&pair).cloned().unwrap_or_default();
        for sequence in sequences_with_pair {
            let mut c = self.cursor_mut(sequence.sequence_index as usize);
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

    /// Retrieve the counts of each pair of tokens in this shard.
    /// E.g. if we have 'l', 'o', 'w' [pretend these are token ids],
    /// should return ('l', 'o) -> 1, ('o', 'w') -> 1
    pub fn counts(&self) -> Vec<CountInfo> {
        let mut ret = Vec::new();

        for (k, v) in self.count_index.iter() {
            let dup_count: u32 = v.iter().fold(0, |acc, e| {
                acc + (e.num_occurrences * self.duplicates[e.sequence_index as usize])
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

    fn update_counts(&mut self, sequence_idx: u32) {
        let c = self.cursor_mut(usize::try_from(sequence_idx).expect("sequence should fit in u32"));
        let pairs = c.into_pairs();
        for pair in pairs.into_iter() {
            self.count_index
                .entry(pair)
                .and_modify(|e| {
                    let pos = e.iter().position(|cv| cv.sequence_index == sequence_idx);
                    match pos {
                        Some(pos) => e[pos].num_occurrences += 1,
                        None => e.push(CountVal {
                            sequence_index: sequence_idx,
                            num_occurrences: 1,
                        }),
                    }
                })
                .or_insert_with(|| {
                    vec![CountVal {
                        sequence_index: sequence_idx,
                        num_occurrences: 1,
                    }]
                });
        }
    }
}

impl Default for SequenceShard {
    fn default() -> Self {
        Self {
            sequences: Vec::new(),
            start_index: Vec::new(),
            duplicates: Vec::new(),
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
    fn to_pairs(c: &mut SequenceCursor) -> Vec<(u32, u32)> {
        let mut ret = Vec::new();

        while !c.is_done() {
            ret.push(c.current_pair().expect("expect a valid pair when not done"));
            c.next();
        }

        ret
    }

    #[test]
    pub fn test_cursor_pair() {
        let mut s = SequenceShard::new();

        s.push(&[1, 1, 2, 3], 2);

        let mut c = s.cursor_mut(0);
        assert_eq!(to_pairs(&mut c), vec![(1, 1), (1, 2), (2, 3)]);
    }

    #[test]
    pub fn test_merge_simpler() {
        let mut s = SequenceShard::new();

        s.push(&[3, 1, 1, 2], 2);
        s.merge_pair((1, 1), 4);

        // now should look like [3, 4, 2]
        let counts = s.counts();

        let mut three_four_found = false;
        let mut four_two_found = false;
        assert_eq!(counts.len(), 2);

        for count in counts {
            if count.token_pair == (3, 4) {
                assert_eq!(count.count, 2);
                three_four_found = true;
            } else if count.token_pair == (4, 2) {
                assert_eq!(count.count, 2);
                four_two_found = true;
            } else {
                assert!(
                    false,
                    "Unexpected token pair ({}, {})",
                    count.token_pair.0, count.token_pair.1
                );
            }
        }

        assert!(three_four_found, "Didn't find (3,4) pair");
        assert!(four_two_found, "Didn't find (4,2) pair");
    }

    #[test]
    pub fn test_merge_runs() {
        let mut s = SequenceShard::new();

        s.push(&[1, 1, 1, 1, 1], 2);
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
                assert_eq!(count.count, 1);
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
}
