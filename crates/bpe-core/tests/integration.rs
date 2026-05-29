static SIMPLE_CORPUS: &[u8] =
    b"low lower lowest new newer newest low new low new the newest lower new low <|endoftext|>";

use bpe_core::tokenize;

struct NoOpInterrupt {}

impl bpe_core::Interrupt for NoOpInterrupt {
    fn check(&self) -> Result<(), bpe_core::BpeError> {
        Ok(())
    }
}

#[test]
fn test_simple_corpus() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let r = tokenize(SIMPLE_CORPUS, 258, special_tokens, NoOpInterrupt {});
    assert!(r.is_ok(), "should be able to tokenize");
}
