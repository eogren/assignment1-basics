use std::{
    cmp::min,
    fmt::Debug,
    fs::File,
    io::{Error, ErrorKind},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use itertools::Itertools;
use memchr::memmem;
use memmap2::Mmap;
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    pretok::{pretokenize_chunk, SequenceBuilder},
    sequence::{CountInfo, SequenceShard},
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

pub trait Interrupt {
    fn check(&self) -> Result<(), BpeError>;
}

#[tracing::instrument(skip(interrupt_fn))]
pub fn tokenize(
    path: PathBuf,
    num_tokens: u32,
    special_tokens: Vec<String>,
    interrupt_fn: impl Interrupt + Send + std::marker::Sync,
) -> Result<(), BpeError> {
    if num_tokens < 256 {
        return Err(BpeError::VocabTooSmall);
    }

    let m = open_file(path)?;
    let chunks = pretokenize(&m, &special_tokens, interrupt_fn)?;
    println!(
        "Parallelized: got {} total sequences in all chunks",
        chunks.counts().len()
    );

    // For now, one sequence chunk
    let mut shard = SequenceShard::new();
    for (k, v) in chunks.counts() {
        let tokens: Vec<u32> = k.iter().copied().map(u32::from).collect();
        shard.push(
            &tokens,
            u32::try_from(*v).expect("size should fit into u32"),
        );
    }

    let counts = shard.counts();
    let biggest_pair = counts.par_iter().reduce(
        || &CountInfo {
            token_pair: (0, 0),
            count: 0,
        },
        |s1, s2| {
            if s1.count > s2.count {
                s1
            } else {
                s2
            }
        },
    );

    println!(
        "Most common pair is ({}, {}) with {} occurrences",
        biggest_pair.token_pair.0, biggest_pair.token_pair.1, biggest_pair.count
    );
    Ok(())
}

#[tracing::instrument(skip(m, interrupt_fn))]
fn pretokenize(
    m: &Mmap,
    special_tokens: &Vec<String>,
    interrupt_fn: impl Interrupt + Send + std::marker::Sync,
) -> Result<SequenceBuilder, BpeError> {
    // For now just use the first special token
    let special_tokens_str = special_tokens.iter().map(|s| s.as_str()).collect_vec();
    let first_token = special_tokens
        .first()
        .map(|s| s.as_bytes())
        .ok_or(BpeError::SpecialTokensRequired)?;

    let chunks = chunk_with_readahead(&m, first_token, 1024 * 1024, 4096);

    let stop_signal = Arc::new(AtomicBool::new(false));
    let watchdog_stop_signal = stop_signal.clone();

    let chunks = std::thread::scope(|s| {
        thread::Builder::new()
            .name("python-watchdog".to_string())
            .spawn_scoped(s, || {
                while !watchdog_stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                    if interrupt_fn.check().is_err() {
                        watchdog_stop_signal.store(true, Ordering::Relaxed);
                    }

                    std::thread::sleep(Duration::from_millis(500));
                }
            })
            .expect("spawn should succeed");

        let merged_chunks = chunks
            .par_iter()
            .map(|chunk| {
                if stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                    None
                } else {
                    Some(pretokenize_chunk(chunk, &special_tokens_str))
                }
            })
            .try_reduce(SequenceBuilder::new, |s1, s2| {
                Some(SequenceBuilder::merge(s1, s2))
            });

        // Stop watchdog
        stop_signal.store(true, Ordering::Relaxed);
        merged_chunks
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

#[cfg(test)]
mod tests {
    use super::*;

    const SEP: &[u8] = b" ";
    const CHUNK: &[u8] = b"Hello World ";

    struct NoOpInterrupt {}

    impl Interrupt for NoOpInterrupt {
        fn check(&self) -> Result<(), BpeError> {
            Ok(())
        }
    }

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
        let i = NoOpInterrupt {};
        let e = tokenize("/tmp".into(), 500, vec!["<|endoftext|>".to_string()], i);
        assert!(matches!(e, Err(BpeError::IoError(_))));
    }

    #[test]
    fn test_tokenize_no_tokens() {
        let i = NoOpInterrupt {};
        let file = tempfile::NamedTempFile::new().expect("failed to create file");
        let e = tokenize(file.path().to_path_buf(), 500, vec![], i);
        assert!(matches!(e, Err(BpeError::SpecialTokensRequired)));
    }
}
