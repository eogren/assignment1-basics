use bpe_core::BpeError::{SpecialTokensRequired, VocabTooSmall};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

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

struct PythonInterrupt {}

fn pyerr_to_bpe(e: &PyErr) -> bpe_core::BpeError {
    let io_err = std::io::Error::other(e.to_string());
    bpe_core::BpeError::IoError(io_err)
}

impl bpe_core::Interrupt for PythonInterrupt {
    fn check(&self) -> Result<(), bpe_core::BpeError> {
        Python::attach(|py| {
            let r = py.check_signals();
            r.map_err(|e| pyerr_to_bpe(&e))
        })
    }
}

#[pyfunction]
fn tokenize(
    py: Python<'_>,
    path: PathBuf,
    vocab_size: u32,
    special_tokens: Vec<String>,
) -> PyResult<(
    HashMap<u32, Vec<u8>>,   /* vocab int->bytes */
    Vec<(Vec<u8>, Vec<u8>)>, /* merge list */
)> {
    let i = PythonInterrupt {};

    let ret = py.detach(move || {
        bpe_core::tokenize_file(path, vocab_size, special_tokens, i).map_err(PyBpeError::from)
    })?;

    Ok(ret)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn bpe_token(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;

    Ok(())
}
