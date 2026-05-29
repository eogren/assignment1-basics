struct NoOpInterrupt {}

impl bpe_core::Interrupt for NoOpInterrupt {
    fn check(&self) -> Result<(), bpe_core::BpeError> {
        Ok(())
    }
}

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/TinyStoriesV2-GPT4-train.txt");
    println!("Tokenizing {}", path.to_str().unwrap_or_default());

    let tokens = vec!["|<endoftext>|".to_string()];
    let r = bpe_core::tokenize_file(path, 10000, tokens, NoOpInterrupt {});
    println!("{:?}", &r);
}
