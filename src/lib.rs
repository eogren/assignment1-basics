use memchr::memmem;
use memmap2::Mmap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::min;
use std::collections::HashMap;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::path::PathBuf;

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

    let chunks = chunk(&mmap, b"<|endoftext|>", 16 * 1024 * 1024);
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

