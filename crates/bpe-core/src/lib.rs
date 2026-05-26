use std::{
    cmp::min,
    fs::File,
    io::{Error, ErrorKind},
    path::PathBuf,
};

use memchr::memmem;
use memmap2::Mmap;
use thiserror::Error;

mod pretok;

#[derive(Error, Debug)]
pub enum BpeError {
    #[error("at least one special token is required")]
    SpecialTokensRequired,

    #[error("I/O error")]
    IoError(#[from] std::io::Error),
}

pub fn tokenize(
    path: PathBuf,
    num_tokens: u64,
    special_tokens: Vec<String>,
) -> Result<(), BpeError> {
    let m = open_file(path)?;

    // For now just use the first special token
    let first_token = special_tokens
        .first()
        .ok_or(BpeError::SpecialTokensRequired)?;

    Ok(())
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

fn chunk<'a>(file: &'a [u8], split_token: &[u8], estimated_chunk_size: usize) -> Vec<&'a [u8]> {
    chunk_with_readahead(file, split_token, estimated_chunk_size, 4096)
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

    #[test]
    fn test_tiny_chunk() {
        let tiny_chunk = chunk(CHUNK, SEP, 1);
        assert_eq!(
            tiny_chunk,
            vec![b"Hello ", b"World "],
            "small chunk size should split appropriately"
        );
    }

    #[test]
    fn test_big_chunk() {
        let big_chunk = chunk(CHUNK, SEP, 64);
        assert_eq!(
            big_chunk,
            vec![b"Hello World "],
            "big chunk should just return the whole thing"
        );
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
        let e = tokenize("/tmp".into(), 500, vec!["<|endoftext|>".to_string()]);
        assert!(matches!(e, Err(BpeError::IoError(_))));
    }

    #[test]
    fn test_tokenize_no_tokens() {
        let file = tempfile::NamedTempFile::new().expect("failed to create file");
        let e = tokenize(file.path().to_path_buf(), 500, vec![]);
        assert!(matches!(e, Err(BpeError::SpecialTokensRequired)));
    }
}
