use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::path::PathBuf;

use bpe_core;

fn open_file(path: PathBuf) -> Result<File, std::io::Error> {
    let f = File::open(path)?;
    let md = f.metadata()?;

    if !md.is_file() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "must pass a valid file",
        ));
    }

    Ok(f)
}

#[pyfunction]
fn tokenize(
    path: PathBuf,
    vocab_size: u32,
    special_tokens: Vec<String>,
) -> PyResult<(
    HashMap<u32, Vec<u8>>,   /* vocab int->bytes */
    Vec<(Vec<u8>, Vec<u8>)>, /* merge list */
)> {
    if vocab_size < 256 {
        return Err(PyErr::new::<PyValueError, _>("vocab_size must be >= 256"));
    }

    let f = open_file(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&f) }?;

    let chunks = bpe_core::chunk(&mmap, b"<|endoftext|>", 16 * 1024 * 1024);
    println!("Generated {} chunks", chunks.len());
    Ok((HashMap::new(), Vec::new()))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn bpe_token(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use bpe_core::chunk_with_readahead;
    use memmap2::Mmap;
    use tempfile::NamedTempFile;

    use super::*;

    const SEP: &[u8] = b" ";

    fn prep_file() -> (NamedTempFile, Mmap) {
        let contents = b"Hello World ";
        let mut file = tempfile::NamedTempFile::new().expect("failed to create file");
        file.write_all(contents)
            .expect("should be able to write to temp file");
        file.flush().expect("should be able to flush temp file");

        let f = open_file(file.path().to_path_buf()).expect("should be able to open temp file");
        let mmap = unsafe { Mmap::map(&f) }.expect("should be able to mmap file");

        (file, mmap)
    }

    #[test]
    fn test_tiny_with_tiny_readahead() {
        let (_f, mmap) = prep_file();
        let tiny_chunk = chunk_with_readahead(&mmap, SEP, 1, 1);
        assert_eq!(
            tiny_chunk,
            vec![b"Hello ", b"World "],
            "small chunk size should split appropriately"
        );
    }
}
