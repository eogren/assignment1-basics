fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/TinyStoriesV2-GPT4-train.txt");
    println!("Tokenizing {}", path.to_str().unwrap_or_default());

    let tokens = vec!["|<endoftext>|".to_string()];
    let r = bpe_core::train_tokenizer_file(path, 10000, tokens, None);
    println!("{:?}", &r);
}
