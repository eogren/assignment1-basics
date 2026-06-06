use bpe_core::BpeError::{SpecialTokensRequired, VocabTooSmall};
use bpe_core::ProgressInfo;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

struct PyBpeError(bpe_core::BpeError);

impl From<bpe_core::BpeError> for PyBpeError {
    fn from(value: bpe_core::BpeError) -> Self {
        PyBpeError(value)
    }
}

impl From<PyBpeError> for PyErr {
    fn from(value: PyBpeError) -> Self {
        match value.0 {
            SpecialTokensRequired => PyValueError::new_err(value.0.to_string()),
            VocabTooSmall => PyValueError::new_err(value.0.to_string()),
            bpe_core::BpeError::IoError(e) => e.into(),
        }
    }
}

#[pyclass(frozen)]
#[derive(Default)]
struct ProgressHandler {
    progress: Arc<ProgressInfo>,
}

#[pyclass]
#[derive(Debug, PartialEq, Eq)]
struct Progress {
    #[pyo3(get)]
    pretoken_total_shards: u32,

    #[pyo3(get)]
    pretoken_done_shards: u32,

    #[pyo3(get)]
    tokenizer_merges_done: u32,
}

impl Progress {
    fn new(pi: &ProgressInfo) -> Self {
        Self {
            pretoken_total_shards: pi.pretoken_total_shards.load(Relaxed),
            pretoken_done_shards: pi.pretoken_done_shards.load(Relaxed),
            tokenizer_merges_done: pi.tokenizer_merges_done.load(Relaxed),
        }
    }
}

#[pymethods]
impl ProgressHandler {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn values(slf: &Bound<'_, Self>) -> Progress {
        Progress::new(&slf.get().progress)
    }
}

#[pyfunction]
#[pyo3(signature = (path, vocab_size, special_tokens, progress_handler = None))]
fn tokenize(
    py: Python<'_>,
    path: PathBuf,
    vocab_size: u32,
    special_tokens: Vec<String>,
    progress_handler: Option<Py<ProgressHandler>>,
) -> PyResult<(
    HashMap<u32, Vec<u8>>,   /* vocab int->bytes */
    Vec<(Vec<u8>, Vec<u8>)>, /* merge list */
)> {
    let stop = Arc::new(AtomicBool::new(false));

    let ret = py.detach(move || {
        bpe_core::tokenize_file(
            path,
            vocab_size,
            special_tokens,
            progress_handler.map(|p| p.get().progress.clone()),
        )
        .map_err(PyBpeError::from)
    });

    stop.store(true, Relaxed);

    ret.map_err(|e| e.into())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn bpe_token(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_class::<Progress>()?;
    m.add_class::<ProgressHandler>()?;
    Ok(())
}
