static SIMPLE_CORPUS: &[u8] =
    b"low lower lowest new newer newest low new low new the newest lower new low <|endoftext|>";

use std::sync::{Arc, atomic::Ordering::Relaxed};

use bpe_core::{ProgressInfo, tokenize};

#[test]
fn test_simple_corpus() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let pi = Arc::new(ProgressInfo::default());

    let r = tokenize(SIMPLE_CORPUS, 258, special_tokens, Some(pi.clone()));
    assert!(r.is_ok(), "should be able to tokenize");
    assert!(pi.tokenizer_merges_done.load(Relaxed) == 258);
}
