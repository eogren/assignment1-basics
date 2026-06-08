use std::{
    cmp::{
        Ordering::{Greater, Less},
        min,
    },
    collections::HashMap,
    fmt::Debug,
    fs::File,
    io::{Error, ErrorKind},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
    },
};

use itertools::Itertools;
use memchr::memmem;
use memmap2::Mmap;
use rayon::prelude::*;
use thiserror::Error;
use tracing::{debug, info};

use crate::{
    pretok::{SequenceBuilder, pretokenize_chunk_for_training},
    sequence::{CountInfo, RealStatsCollector, SequenceShard},
};

mod pretok;
mod sequence;

#[derive(Error, Debug)]
pub enum BpeError {
    #[error("at least one special token is required")]
    SpecialTokensRequired,

    #[error("not enough tokens in vocab. we start with at least 256")]
    VocabTooSmall,

    #[error("I/O error")]
    IoError(#[from] std::io::Error),
}

/// Keeps track of the progress of the tokenize() function.
#[derive(Debug, Default)]
pub struct ProgressInfo {
    pub pretoken_unique_sequences: AtomicU32,

    pub pretoken_total_shards: AtomicU32,
    pub pretoken_done_shards: AtomicU32,
    pub tokenizer_merges_done: AtomicU32,

    pub should_stop: AtomicBool,
}

#[tracing::instrument(skip(progress_info))]
pub fn train_tokenizer_file(
    path: PathBuf,
    num_tokens: u32,
    special_tokens: Vec<String>,
    progress_info: Option<Arc<ProgressInfo>>,
) -> Result<(HashMap<u32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), BpeError> {
    let m = open_file(path)?;
    train_tokenizer(&m, num_tokens, special_tokens, progress_info)
}

pub fn train_tokenizer(
    buf: &[u8],
    num_tokens: u32,
    special_tokens: Vec<String>,
    progress_info: Option<Arc<ProgressInfo>>,
) -> Result<(HashMap<u32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), BpeError> {
    if num_tokens
        < 256 + u32::try_from(special_tokens.len()).expect("special tokens should fit into u32")
    {
        return Err(BpeError::VocabTooSmall);
    }

    let mut token_dict = starting_token_dict(&special_tokens);
    let mut merge_list = Vec::new();

    if let Some(ref pi) = progress_info {
        pi.tokenizer_merges_done.store(
            u32::try_from(token_dict.len()).expect("merges should fit in u32"),
            Relaxed,
        );
    }

    let chunks = pretokenize_for_training(buf, &special_tokens, progress_info.clone())?;
    if let Some(ref pi) = progress_info {
        pi.pretoken_unique_sequences.store(
            u32::try_from(chunks.counts().len()).expect("sequence count should fit in u32"),
            Relaxed,
        );
    }

    let mut shard = generate_sequence_shards_with_stats(chunks);

    info!("Starting merge passes");
    while token_dict.len() < num_tokens as usize {
        let biggest_pair = most_frequent_token(&token_dict, &shard);

        let new_token_id = u32::try_from(token_dict.len()).expect("tokens should fit in u32");

        shard.merge_pair(biggest_pair, new_token_id);

        let new_token = token_dict[&biggest_pair.0]
            .iter()
            .chain(token_dict[&biggest_pair.1].iter())
            .copied()
            .collect_vec();

        let new_merge = (
            token_dict[&biggest_pair.0].clone(),
            token_dict[&biggest_pair.1].clone(),
        );

        debug!(
            "Merging ({}, {}) ('{:?}', '{:?}') into token {} ('{:?}')",
            &biggest_pair.0, &biggest_pair.1, &new_merge.0, &new_merge.1, &new_token_id, &new_token,
        );
        merge_list.push(new_merge);
        assert_eq!(
            token_dict.insert(new_token_id, new_token),
            None,
            "should never be replacing a token id"
        );

        debug_assert!(
            merge_list
                .iter()
                .filter(|x| x.0 == token_dict[&biggest_pair.0]
                    && x.1 == token_dict[&biggest_pair.1])
                .count()
                == 1,
            "should never have dupes in merge list"
        );

        if let Some(ref pi) = progress_info {
            pi.tokenizer_merges_done.fetch_add(1, Relaxed);
        }
    }
    Ok((token_dict, merge_list))
}

fn most_frequent_token(
    token_dict: &HashMap<u32, Vec<u8>>,
    shard: &SequenceShard<RealStatsCollector>,
) -> (u32, u32) {
    let counts = shard.counts();
    let biggest_pair = counts.par_iter().reduce(
        || &CountInfo {
            token_pair: (0, 0),
            count: 0,
        },
        |s1, s2| match s1.compare_to(s2, token_dict) {
            Greater => s1,
            Less => s2,
            _ => panic!("Unexpected result from comparing CountInfos"),
        },
    );
    biggest_pair.token_pair
}

#[tracing::instrument(skip(chunks))]
fn generate_sequence_shards_with_stats(
    chunks: SequenceBuilder,
) -> SequenceShard<RealStatsCollector> {
    info!(
        "Generating pair sequences from {} unique sequences from corpus",
        chunks.counts().len()
    );

    let mut count = 0;

    // For now, one sequence chunk
    let mut shard = SequenceShard::new(RealStatsCollector::default());
    for (k, v) in chunks.counts() {
        let tokens: Vec<u32> = k.iter().copied().map(u32::from).collect();
        shard.push(
            &tokens,
            u32::try_from(*v).expect("size should fit into u32"),
        );
        count += 1;

        if count % 1000 == 0 {
            debug!("{} pairs generated", count);
        }
    }
    shard
}

#[tracing::instrument(skip(m, progress_info))]
fn pretokenize_for_training(
    m: &[u8],
    special_tokens: &Vec<String>,
    progress_info: Option<Arc<ProgressInfo>>,
) -> Result<SequenceBuilder, BpeError> {
    // For now just use the first special token
    let special_tokens_str = special_tokens.iter().map(|s| s.as_str()).collect_vec();
    let first_token = special_tokens
        .first()
        .map(|s| s.as_bytes())
        .ok_or(BpeError::SpecialTokensRequired)?;

    let chunks = chunk_with_readahead(m, first_token, 1024 * 1024, 4096);

    debug!("Pretokenize: {} chunks to process", chunks.len());

    if let Some(ref pi) = progress_info {
        pi.pretoken_total_shards.store(
            u32::try_from(chunks.len()).expect("chunks should fit in u32"),
            Relaxed,
        );
    }
    let watchdog_stop_signal = progress_info;

    let chunks = std::thread::scope(|_s| {
        chunks
            .par_iter()
            .map(|chunk| {
                if let Some(watchdog_pi) = &watchdog_stop_signal
                    && watchdog_pi
                        .should_stop
                        .load(std::sync::atomic::Ordering::Relaxed)
                {
                    None
                } else {
                    let ret = Some(pretokenize_chunk_for_training(chunk, &special_tokens_str));
                    if let Some(watchdog_pi) = &watchdog_stop_signal {
                        watchdog_pi.pretoken_done_shards.fetch_add(1, Relaxed);
                    }

                    ret
                }
            })
            .try_reduce(SequenceBuilder::new, |s1, s2| {
                Some(SequenceBuilder::merge(s1, s2))
            })
    })
    .expect("should be able to retrieve chunks");

    Ok(chunks)
}

/// Chunk the given file with approx `estimated_chunk_size` chunks.
fn chunk_with_readahead<'a>(
    file: &'a [u8],
    split_token: &[u8],
    estimated_chunk_size: usize,
    readahead: usize,
) -> Vec<&'a [u8]> {
    assert!(readahead > 0, "Readahead must be larger than zero");

    let mut start_idx = 0;
    let mut ret = Vec::new();

    while start_idx < file.len() {
        let mut end_idx = min(file.len(), start_idx + estimated_chunk_size);
        let mut next_sep_idx = None;

        while next_sep_idx.is_none() {
            let end_readahead_idx = min(end_idx + readahead, file.len());
            let readahead = &file[end_idx..end_readahead_idx];

            if readahead.is_empty() {
                next_sep_idx = Some(file.len());
            } else {
                next_sep_idx = memmem::find(readahead, split_token)
                    .map(|n| min(file.len(), end_idx + n + split_token.len()));
                end_idx = end_readahead_idx;
            }
        }

        ret.push(&file[start_idx..next_sep_idx.expect("next_sep_idx should always be valid here")]);

        start_idx = next_sep_idx.expect("next_sep_idx should be valid");
    }

    ret
}

fn open_file(path: PathBuf) -> Result<Mmap, std::io::Error> {
    let f = File::open(path)?;
    let md = f.metadata()?;

    if !md.is_file() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "must pass a valid file",
        ));
    }

    let mmap = unsafe { memmap2::Mmap::map(&f) }?;

    Ok(mmap)
}

fn starting_token_dict(tokens: &Vec<String>) -> HashMap<u32, Vec<u8>> {
    let mut ret = HashMap::new();

    for i in 0..256_u32 {
        ret.insert(i, vec![u8::try_from(i).expect("0-255 should be in u8")]);
    }

    for token in tokens {
        let bytes = token.bytes().collect_vec();
        ret.insert(
            u32::try_from(ret.len()).expect("token should fit in u32"),
            bytes,
        );
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEP: &[u8] = b" ";
    const CHUNK: &[u8] = b"Hello World ";

    #[test]
    fn test_tiny_with_tiny_readahead() {
        let tiny_chunk = chunk_with_readahead(CHUNK, SEP, 1, 1);
        assert_eq!(
            tiny_chunk,
            vec![b"Hello ", b"World "],
            "small chunk size should split appropriately"
        );
    }

    #[test]
    fn test_tokenize_err_on_invalid_file() {
        let e = train_tokenizer_file("/tmp".into(), 500, vec!["<|endoftext|>".to_string()], None);
        assert!(matches!(e, Err(BpeError::IoError(_))));
    }

    #[test]
    fn test_tokenize_no_tokens() {
        let file = tempfile::NamedTempFile::new().expect("failed to create file");
        let e = train_tokenizer_file(file.path().to_path_buf(), 500, vec![], None);
        assert!(matches!(e, Err(BpeError::SpecialTokensRequired)));
    }

    // todo check progress
}
