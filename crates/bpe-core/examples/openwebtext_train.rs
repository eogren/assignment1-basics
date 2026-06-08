use std::sync::{Arc, atomic::Ordering::Relaxed};

use bpe_core::ProgressInfo;
use tracing::info;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self, format::FmtSpan},
    prelude::*,
};

fn main() {
    //let (chrome_layer, _guard) = ChromeLayerBuilder::new().build();

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
        .with(fmt::layer().with_span_events(FmtSpan::CLOSE)) // <- stdout, with span timing
        .init();

    let pi = Arc::new(ProgressInfo::default());
    let handler_pi = pi.clone();
    ctrlc::set_handler(move || {
        info!("Ctrl+C received! Flushing Chrome trace and exiting...");
        handler_pi.should_stop.store(true, Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/owt_train.txt");
    println!("Tokenizing {}", path.to_str().unwrap_or_default());

    let tokens = vec!["<|endoftext|>".to_string()];
    let r = bpe_core::train_tokenizer_file(path, 30000, tokens, Some(pi));
    println!("{:?}", &r);
}
